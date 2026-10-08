from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import argparse
import torch
import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import TabularDataset, load_config, dump_config


def dp_histogram(labels, num_classes, epsilon=0.1, delta=1e-5):

    labels = torch.tensor(labels.values, dtype=torch.long).view(-1)

    counts = torch.bincount(labels, minlength=num_classes).float()

    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    noise = torch.normal(0, sigma, size=counts.shape)
    noisy_counts = counts + noise

    noisy_counts = torch.clamp(noisy_counts, min=1e-6)

    p_y = noisy_counts / noisy_counts.sum()

    return p_y.numpy()


# class RDPAccountant:
#     def __init__(self, sample_rate, steps):
#         self.q = sample_rate
#         self.steps = steps

#     def rdp_get_noise_multiplier(self, epsilon, delta):

#         noise_multiplier = get_noise_multiplier(
#                         target_epsilon=epsilon,
#                         target_delta=delta,
#                         sample_rate=self.q,
#                         steps=self.steps,
#                         accountant='rdp',
#                     )
            
#         return noise_multiplier


class Accountant:
    def __init__(self, sample_rate, steps):
        self.q = sample_rate
        self.steps = steps

    def rdp_get_noise_multiplier(self, epsilon, delta):
        noise_multiplier = get_noise_multiplier(
                        target_epsilon=epsilon,
                        target_delta=delta,
                        sample_rate=self.q,
                        steps=self.steps,
                        accountant='rdp',
                    )
            
        return noise_multiplier

    def gdp_get_mu(self, noise_multiplier):
        sigma = noise_multiplier

        return self.q * np.sqrt(
            self.steps * (np.exp(1.0 / sigma**2) - 1.0)
        )

    @staticmethod
    def gdp_delta_from_mu(epsilon, mu):
        if mu <= 0:
            return 0.0

        return (
            norm.cdf(mu / 2 - epsilon / mu)
            - np.exp(epsilon)
            * norm.cdf(-mu / 2 - epsilon / mu)
        )

    def gdp_get_epsilon(self, noise_multiplier, delta):
        mu = self.gdp_get_mu(noise_multiplier)

        def func(epsilon):
            return self.gdp_delta_from_mu(epsilon, mu) - delta

        return brentq(func, 0.0, 1000.0)

    def gdp_get_noise_multiplier(self, epsilon, delta):

        def func(sigma):
            mu = self.gdp_get_mu(sigma)

            delta1 = self.gdp_delta_from_mu(
                epsilon,
                mu,
            )

            return delta1 - delta

        return brentq(func, 0.1, 100.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default='configs/adult/config.toml')
    args = parser.parse_args()
    raw_config = load_config(args.config)
    exp_path = raw_config['exp_path']
    batch_size = raw_config['train']['main']['batch_size']
    epochs = raw_config['train']['main']['epochs']
    dataset = TabularDataset(exp_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    sample_rate = 1 / len(train_loader)
    steps = int(epochs / sample_rate)

    accountant = Accountant(sample_rate, steps)

    noise = accountant.gdp_get_noise_multiplier(epsilon=10, delta=1e-5)

    raw_config['dp']['sigma'] = noise

    dump_config(raw_config, args.config)
