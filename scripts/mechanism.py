from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import argparse
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import TabularDataset, load_config


def mechanism(
        exp_path='exp/adult',
        epochs=100,
        batch_size=256,
        target_epsilon=10,
        target_delta=1e-5,
):

    dataset = TabularDataset(exp_path)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)

    sample_rate = 1 / len(train_loader)

    noise_multiplier = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant='prv',
            )
    
    print(noise_multiplier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='configs/adult/config.toml')
    args = parser.parse_args()
    raw_config = load_config(args.config)
    mechanism(
        exp_path=raw_config['exp_path'], 
        epochs=raw_config['train']['main']['epochs'], 
        batch_size=raw_config['train']['main']['batch_size'], 
        target_epsilon=raw_config['dp']['epsilon']
    )
