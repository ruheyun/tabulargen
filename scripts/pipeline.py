import json
import os
import argparse
import torch
from preprocess import data_process
from dm_train import train
from dm_sample import sample
from eval_catboost import train_catboost
from eval_simple import train_simple
from utils import load_config
import warnings
warnings.filterwarnings('ignore')


def save_config(exp_dir, config):
    os.makedirs(exp_dir, exist_ok=True)
    filepath = os.path.join(exp_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    # 启用训练、采样、测试
    parser.add_argument('--config', metavar='FILE', default='configs/adult/config.toml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    # 评估模型设置
    parser.add_argument('--eval_model', type=str, choices=['catboost', 'simple'], default='catboost')

    args = parser.parse_args()
    raw_config = load_config(args.config)

    if args.eval_model != 'catboost':
        raw_config['eval']['type']['eval_model'] = args.eval_model

    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cpu')

    save_config(raw_config['exp_path'], raw_config)

    data_process(data_path=raw_config['data_path'], exp_path=raw_config['exp_path'], num_encoder='quantile', cat_encoder='alb')


    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            exp_path=raw_config['exp_path'],
            model_params=raw_config['model_params'],
            dp_params=raw_config['dp'],
            device=device,
        )

    if args.sample:
        sample(
            **raw_config['diffusion_params'],
            exp_path=raw_config['exp_path'],
            batch_size=raw_config['sample']['batch_size'],
            num_samples=raw_config['sample']['num_samples'],
            model_path=os.path.join(raw_config['exp_path'], 'model_ema.pt'),
            model_params=raw_config['model_params'],
            device=device,
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


if __name__ == '__main__':
    main()
