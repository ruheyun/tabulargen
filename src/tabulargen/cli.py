"""Run explicit tabular encoding, training, sampling, and evaluation stages."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tabulargen.artifacts import ENCODED_FILES, require_files
from tabulargen.config import resolve_config
from tabulargen.checkpoints import load_checkpoint
from tabulargen.layout import RunPaths


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, stages, checkpoint_path=None):
    """Execute stages without changing the caller's resolved configuration."""
    config = deepcopy(config)
    paths = RunPaths(Path(config['experiment']['path']))
    data_path = config['data']['path']
    checkpoint_path = Path(checkpoint_path or paths.checkpoints / 'checkpoint.pt').resolve()
    sample_dir = paths.samples(config['sample']['seed'])

    if 'encode' in stages:
        from tabulargen.data.preprocess import data_process
        if (paths.checkpoints / 'checkpoint.pt').exists():
            raise FileExistsError('This training run already exists; choose a new experiment.path')
        data_process(data_path, str(paths.encoded), **config['encoding'])
        write_json(paths.encoded / 'config.json', {'data': config['data'], 'encoding': config['encoding']})

    if 'train' in stages:
        import torch
        from tabulargen.training.trainer import train
        require_files(paths.encoded, ENCODED_FILES, 'Training (run --encode first)')
        # Do not permit a configuration to silently label a different encoding.
        encoding_info = json.loads((paths.encoded / 'info.json').read_text())
        if encoding_info.get('encoding') != config['encoding']:
            raise ValueError('Encoding configuration differs from encoded artifacts; start a new run')
        train(exp_path=str(paths.encoded), checkpoint_path=paths.checkpoints, log_path=paths.logs,
              model_params=config['model'], dp_params=config['privacy'],
              device=torch.device(config['device']), **config['diffusion'], **config['train'])
        write_json(paths.root / 'config.json', config)

    if 'sample' in stages:
        import torch
        from tabulargen.sampling.sampler import sample
        require_files(checkpoint_path.parent, [checkpoint_path.name], 'Sampling (run --train first)')
        if (sample_dir / 'reverse.csv').exists():
            raise FileExistsError('This sample seed already exists; choose another seed or run')
        settings = config['sample']
        encoded = sample(exp_path=str(sample_dir), model_path=checkpoint_path,
               batch_size=settings['batch_size'], num_samples=settings['num_samples'],
               seed=settings['seed'], device=torch.device(config['device']),
               disbalance='uniform' if settings['class_distribution'] == 'uniform' else None)
        write_json(sample_dir / 'config.json', {
            'schema_version': 2, 'sample': settings,
            'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_hash(checkpoint_path),
            'encoded_path': str(encoded), 'samples_sha256': file_hash(sample_dir / 'reverse.csv'),
        })

    if 'eval' in stages:
        settings = config['evaluation']
        model, mode, seed = settings['model'], settings['mode'], settings['seed']
        output = paths.evaluation(mode, config['sample']['seed'], model, seed)
        if (output / 'results.json').exists():
            raise FileExistsError('This evaluation already exists; choose another evaluation seed or run')
        encoded = paths.encoded
        inputs = {'real_data': data_path}
        if mode == 'synthetic':
            require_files(sample_dir, ['reverse.csv', 'config.json'], 'Evaluation (run --sample first)')
            provenance = json.loads((sample_dir / 'config.json').read_text())
            # Relative-to-run checkpoints remain valid after moving a run as a unit.
            recorded_checkpoint = Path(provenance['checkpoint'])
            candidate = checkpoint_path if checkpoint_path.exists() else recorded_checkpoint
            if file_hash(candidate) != provenance['checkpoint_sha256']:
                raise ValueError('Samples belong to a different checkpoint')
            _, encoded = load_checkpoint(candidate)
            if file_hash(sample_dir / 'reverse.csv') != provenance['samples_sha256']:
                raise ValueError('Sample data changed since generation')
            inputs.update(samples=str(sample_dir / 'reverse.csv'),
                          samples_sha256=provenance['samples_sha256'],
                          checkpoint_sha256=provenance['checkpoint_sha256'])
        from tabulargen.evaluation.runner import evaluate_models
        params = deepcopy(settings.get('params', {}))
        if model in ('catboost', 'all'):
            catboost_params = params.get('catboost', {}) if model == 'all' else params
            if 'catboost_params_path' in settings:
                catboost_params = json.loads(Path(settings['catboost_params_path']).read_text()) | catboost_params
            if not catboost_params:
                raise ValueError('Provide evaluation.catboost_params_path or CatBoost parameters')
            if model == 'all':
                params['catboost'] = catboost_params
            else:
                params = catboost_params
        if model in ('simple', 'all'):
            require_files(encoded, ENCODED_FILES, 'Evaluation (run --encode first)')
        result = evaluate_models(data_path, str(sample_dir), model=model, seed=seed,
                                 eval_type=mode, params=params, encoded_path=encoded)
        write_json(output / 'results.json', {
            'schema_version': 1, 'model': model, 'mode': mode,
            'sample_seed': config['sample']['seed'] if mode == 'synthetic' else None,
            'evaluation_seed': seed, 'inputs': inputs, **result,
        })
        write_json(output / 'config.json', settings | {'params': params})
        print(f'Results saved to {output / "results.json"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Version 2 TOML configuration')
    for stage in ['encode', 'train', 'sample', 'eval']:
        parser.add_argument('--' + stage, action='store_true')
    parser.add_argument('--sample-seed', type=int)
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--eval-model', choices=['all', 'catboost', 'simple'])
    parser.add_argument('--checkpoint', help='Optional checkpoint path; defaults to this run')
    args = parser.parse_args(argv)
    stages = [s for s in ['encode', 'train', 'sample', 'eval'] if getattr(args, s)]
    if not stages:
        parser.error('Choose at least one stage: --encode, --train, --sample, --eval')
    config = resolve_config(args.config)
    for value, section in [(args.sample_seed, 'sample'), (args.eval_seed, 'evaluation')]:
        if value is not None:
            if not 0 <= value < 2**32:
                parser.error('Seeds must be in [0, 2**32)')
            config[section]['seed'] = value
    if args.eval_model:
        config['evaluation']['model'] = args.eval_model
    run(config, stages, args.checkpoint)

