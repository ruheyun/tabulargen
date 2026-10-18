import json
import argparse
import os
import sys
import warnings
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from scripts.data.preprocess import data_process
from scripts.training.dm_train import train
from scripts.sampling.dm_sample import sample
from scripts.evaluation.eval_tstr import train_tstr
from utils import load_config, RunPaths, average_metrics, print_metrics

warnings.filterwarnings('ignore')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    # 启用编码、训练、采样、测试
    parser.add_argument('--config', metavar='FILE', default='configs/adult/config.toml')
    parser.add_argument('--encode', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval5', action='store_true', default=False)
    parser.add_argument('--sample_seed', type=int)

    args = parser.parse_args()
    raw_config = load_config(args.config)

    if args.sample_seed is not None and args.sample_seed >= 0:
        raw_config['sample']['seed'] = args.sample_seed

    paths = RunPaths(Path(raw_config['experiment']['path']))
    data_path = raw_config['data']['path']
    checkpoint_path = Path(paths.checkpoints / 'checkpoint.pt').resolve()
    sample_dir = paths.samples(raw_config['sample']['seed'])


    if args.encode:
        data_process(data_path, str(paths.encoded), **raw_config['encoding'])


    if args.train:
        train(
            **raw_config['train'],
            **raw_config['diffusion'],
            exp_path=str(paths.encoded),
            checkpoint_path=str(paths.checkpoints),
            log_path=str(paths.logs),
            model_params=raw_config['model'],
            dp_params=raw_config['privacy'],
            device=raw_config['device'],
        )


    if args.sample:
        sample(
            exp_path=str(sample_dir),
            batch_size=raw_config['sample']['batch_size'],
            num_samples=raw_config['sample']['num_samples'],
            model_path=checkpoint_path,
            device=raw_config['device'],
            disbalance='uniform' if raw_config['sample']['class_distribution'] == 'uniform' else None,
            seed=raw_config['sample'].get('seed', 0)
        )


    if args.eval:
        settings = raw_config['evaluation']
        models, mode, seed = settings['models'], settings['mode'], settings['seed']
        output = paths.evaluation(seed)

        per_model = {}

        result = train_tstr(
            data_path,
            str(sample_dir),
            seed=seed,
            eval_type=mode,
            params_path=settings.get('catboost_params_path'),
            model_names=models,
        )
        per_model.update(result['per_model'])

        metrics = average_metrics(per_model)
        print('Average results')
        print_metrics(metrics)

        result = {'metrics': metrics, 'per_model': per_model}

        write_json(output / 'results.json', {
            'models': models, 
            'mode': mode,
            'sample_seed': raw_config['sample']['seed'] if mode == 'synthetic' else None,
            'evaluation_seed': seed, 
            **result,
        })
        print(f'Results saved to {output / "results.json"}')


    if args.eval5:
        settings = raw_config['evaluation']
        models, mode = settings['models'], settings['mode']
        output = paths.evaluation(5)

        per_seed = {}

        for seed in range(5):
            seed_results = {}

            seed_results = train_tstr(
                data_path,
                str(sample_dir),
                seed=seed,
                eval_type=mode,
                params_path=settings.get('catboost_params_path'),
                model_names=models,
            )

            per_seed[seed] = seed_results['per_model']

        per_model = {
            name: average_metrics({
                seed: results[name]
                for seed, results in per_seed.items()
            })
            for name in models
        }

        groups = {}
        if 'catboost' in per_model:
            groups['catboost'] = per_model['catboost']

        simple_results = {
            name: result
            for name, result in per_model.items()
            if name != 'catboost'
        }
        if simple_results:
            groups['simple'] = average_metrics(simple_results)

        metrics = average_metrics(groups)
        print('Average results')
        print_metrics(metrics)

        result = {'metrics': metrics, 'per_model': per_model}

        write_json(output / 'results.json', {
            'models': models, 
            'mode': mode,
            'sample_seed': raw_config['sample']['seed'] if mode == 'synthetic' else None,
            **result,
        })
        print(f'Results saved to {output / "results.json"}')


if __name__ == '__main__':
    main()
