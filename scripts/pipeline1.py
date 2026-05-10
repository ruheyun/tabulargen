import json
import os
import argparse
import torch
from preprocess import data_process
from dm_train import train
from dm_sample import sample
from eval_catboost import train_catboost
from eval_simple import train_simple
from utils import load_config, load_json
import warnings
warnings.filterwarnings('ignore')


def save_config(exp_dir, config):
    os.makedirs(exp_dir, exist_ok=True)
    filepath = os.path.join(exp_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_name', type=str, default='adult')

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.00039806870482992913)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--sample_seed', type=int, default=0)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_model', type=str, choices=['catboost', 'simple'], default='catboost')
    parser.add_argument('--eval_type', type=str, choices=['real', 'synthetic'], default='synthetic')

    parser.add_argument('--dp', action='store_true', default=False)
    parser.add_argument('--epsilon', type=float, default=10)

    args = parser.parse_args()
    info = load_json(os.path.join('data', args.data_name, 'info.json'))

    raw_config = {
        'data_path': os.path.join('data', args.data_name),
        'exp_path': os.path.join('exp', args.data_name),
        'device': torch.device(args.device if torch.cuda.is_available() else 'cpu'),
        'seed': 0,
        'model_params': {
            'is_y_cond': True,
            'num_classes': info['n_classes'],
            'rtdl_params': {
                'd_layers': [512, 1024, 1024, 512],
                'dropout': 0.0
            }
        },
        'diffusion_params': {
            'num_timesteps': 500,
            'gaussian_loss_type': 'mse',
            'scheduler': 'cosine'
        },
        'train': {
            'main': {
                'epochs': 50,
                'lr': args.lr,
                'weight_decay': 0.0,
                'batch_size': 128
            }
        },
        'sample': {
            'num_samples': info['train_size'],
            'batch_size': 256,
            'seed': args.sample_seed
        },
        'eval': {
            'type': {
                'eval_model': args.eval_model,
                'eval_type': args.eval_type
            }
        },
        'dp': {
            'is_dp': args.dp,
            'epsilon': args.epsilon,
            'max_grad_norm': 1.0,
            'delta': 1e-5
        }
    }

    save_config(raw_config['exp_path'], raw_config)

    data_process(raw_config['data_path'], raw_config['exp_path'], num_encoder='minmax', cat_encoder='alb')


    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            exp_path=raw_config['exp_path'],
            model_params=raw_config['model_params'],
            dp_params=raw_config['dp'],
            device=raw_config['device'],
        )

    if args.sample:
        sample(
            **raw_config['diffusion_params'],
            exp_path=raw_config['exp_path'],
            batch_size=raw_config['sample']['batch_size'],
            num_samples=raw_config['sample']['num_samples'],
            model_path=os.path.join(raw_config['exp_path'], 'model_ema.pt'),
            model_params=raw_config['model_params'],
            device=raw_config['device'],
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
