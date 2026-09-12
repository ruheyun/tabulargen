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
from scripts.evaluation.eval_catboost import train_catboost
from scripts.evaluation.eval_simple import train_simple
from utils import load_config, dump_json, RunPaths

warnings.filterwarnings('ignore')


def save_config(exp_dir, config):
    os.makedirs(exp_dir, exist_ok=True)
    filepath = os.path.join(exp_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    # 启用编码、训练、采样、测试
    parser.add_argument('--config', metavar='FILE', default='configs/adult/config.toml')
    parser.add_argument('--encode', action='store_true', default=True)
    parser.add_argument('--train', action='store_true', default=True)
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
            **raw_config['diffusion'],
            exp_path=str(sample_dir),
            batch_size=raw_config['sample']['batch_size'],
            num_samples=raw_config['sample']['num_samples'],
            model_path=checkpoint_path,
            model_params=raw_config['model'],
            device=raw_config['device'],
            disbalance='uniform' if raw_config['sample']['class_distribution'] == 'uniform' else None,
            seed=raw_config['sample'].get('seed', 0)
        )


    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                data_path=raw_config['data_path'],
                exp_path=raw_config['exp_path'],
                seed=raw_config['seed'],
                eval_type=raw_config['eval']['type']['eval_type'],
            )
        
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                data_path=raw_config['data_path'],
                exp_path=raw_config['exp_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                seed=raw_config['seed'],
            )

        else:
            print('No eval model!')

    # dump_json(raw_config['experiment']['path'], raw_config)


if __name__ == '__main__':
    main()
