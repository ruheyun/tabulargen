import json
import argparse
import os
import sys
import warnings
from pathlib import Path
from copy import deepcopy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from scripts.data.preprocess import data_process
from scripts.training.dm_train import train
from scripts.sampling.dm_sample import sample
from scripts.evaluation.eval_catboost import train_catboost
from scripts.evaluation.eval_simple import train_simple
# from scripts.evaluation.evaluation import evaluate_models
from utils import load_config, dump_json, RunPaths, average_metrics, print_metrics

warnings.filterwarnings('ignore')


def save_config(exp_dir, config):
    os.makedirs(exp_dir, exist_ok=True)
    filepath = os.path.join(exp_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

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

        encoded = paths.encoded

        per_model = {}
        for model in models:
            if model == 'catboost':
                catboost_params = json.loads(Path(settings['catboost_params_path']).read_text())
                result = train_catboost(data_path, str(sample_dir), seed=seed, eval_type=mode, params=catboost_params)
                per_model.update(result['per_model'])
            else:
                result = train_simple(data_path, str(sample_dir), seed=seed, eval_type=mode, params=None, encoded_path=encoded)
                per_model.update(result['per_model'])
                break

        metrics = average_metrics(per_model)
        print('Average results')
        print_metrics(metrics)

        result = {'metrics': metrics, 'per_model': per_model}


        # if model in ('catboost', 'all'):
        #     catboost_params = params.get('catboost', {}) if model == 'all' else params
        #     if 'catboost_params_path' in settings:
        #         catboost_params = json.loads(Path(settings['catboost_params_path']).read_text()) | catboost_params
        #     if not catboost_params:
        #         raise ValueError('Provide evaluation.catboost_params_path or CatBoost parameters')
        #     if model == 'all':
        #         params['catboost'] = catboost_params
        #     else:
        #         params = catboost_params
        # result = evaluate_models(data_path, str(sample_dir), model=model, seed=seed,
        #                          eval_type=mode, params=params, encoded_path=encoded)

        write_json(output / 'results.json', {
            'models': models, 
            'mode': mode,
            'sample_seed': raw_config['sample']['seed'] if mode == 'synthetic' else None,
            'evaluation_seed': seed, 
            **result,
        })
        print(f'Results saved to {output / "results.json"}')


if __name__ == '__main__':
    main()
