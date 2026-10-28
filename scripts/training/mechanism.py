from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import argparse
import torch
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import norm
from sklearn.cluster import KMeans
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


def k_mearns(K, lst, X):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

    # 计算分界点
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.sort(centers)  
    print("排序后的簇质心:", sorted_centers)

    # 相邻质心中点作为分界点
    boundaries_noise = (sorted_centers[:-1] + sorted_centers[1:]) / 2

    # 找对应的时间步分界（近似）
    t_boundaries = []
    for b in boundaries_noise:
        idx = np.argmin(np.abs(lst.flatten() - b))
        t_boundaries.append(X[idx])
    t_boundaries = np.array(t_boundaries)
    print("对应的时间步分界点（近似）:", t_boundaries)

    return t_boundaries


def epsilon_allocation(K, type):
    k = np.arange(1, K + 1)

    if type == 'HA':
        # 1. Harmonic Allocation
        ratio = (1/k - 1/(k + 1)) * (K + 1) / K
    elif type == 'EA':
        # 2. Exponential Allocation
        ratio = (2**(K - k)) / (2**K - 1)
    else:
        raise ValueError('type is error')

    return ratio


def k_dp_params(K, lst, X, type, sample_rate, steps, epsilon, delta, account):
    t_bd = k_mearns(K, lst, X)

    rt = epsilon_allocation(K, type)

    noise_multiplier = []
    accountant = Accountant(sample_rate, steps)
    for k in range(K):
        noise_multiplier = accountant.get_noise_multiplier(
                        epsilon=epsilon * rt[k], 
                        delta=delta,
                        account='gdp'
                    )
    return

class Accountant:
    def __init__(self, sample_rate, steps):
        self.q = sample_rate
        self.steps = steps

    def get_noise_multiplier(self, epsilon, delta, account='gdp', max_order=100):
        if account == 'gdp':
            return self.gdp_get_noise_multiplier(epsilon, delta)
        elif account == 'rdp':
            return self.rdp_get_noise_multiplier(epsilon, delta)
        elif account == 'ma':
            return self.ma_get_noise_multiplier(epsilon, delta, max_order)
        else:
            raise ValueError('account is error!')

    def ma_get_epsilon(self, noise_multiplier, delta, max_order=100):
        sigma = noise_multiplier

        epsilons = []
        for lam in range(1, max_order + 1):
            alpha = lam + 1
            i = np.arange(alpha + 1)

            log_binom = (
                gammaln(alpha + 1)
                - gammaln(i + 1)
                - gammaln(alpha - i + 1)
            )

            log_terms = (
                log_binom
                + i * np.log(self.q)
                + (alpha - i) * np.log1p(-self.q)
                + (i**2 - i) / (2 * sigma**2)
            ) if self.q < 1 else np.where(
                i == alpha,
                (i**2 - i) / (2 * sigma**2),
                -np.inf,
            )

            log_a = logsumexp(log_terms)
            log_moment = self.steps * log_a

            eps = (
                log_moment + np.log(1 / delta)
            ) / lam

            epsilons.append(eps)

        return max(0.0, min(epsilons))

    def ma_get_noise_multiplier(self, epsilon, delta, max_order=100):
        def objective(sigma):
            return (
                self.ma_get_epsilon(
                    noise_multiplier=sigma,
                    delta=delta,
                    max_order=max_order
                )
                - epsilon
            )

        noise_multiplier = brentq(objective, 0.01, 1e4)

        return noise_multiplier

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

    def gdp_get_mu_from_epsilon_delta(self, epsilon, delta):
        if epsilon < 0 or delta <= 0 or delta >= 1:
            raise ValueError("epsilon need >= 0, and delta need 0 < delta < 1")

        def func(mu):
            return self.gdp_delta_from_mu(epsilon, mu) - delta

        lo = 0.0
        hi = 1.0
        while func(hi) < 0:
            hi *= 2.0

        return brentq(func, lo, hi)


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

    ma_noise = accountant.ma_get_noise_multiplier(epsilon=1, delta=1e-5)
    rdp_noise = accountant.rdp_get_noise_multiplier(epsilon=1, delta=1e-5)
    gdp_noise = accountant.gdp_get_noise_multiplier(epsilon=1, delta=1e-5)
    print(f'ma_noise: {ma_noise}\nrdp_noise: {rdp_noise}\ngdp_noise: {gdp_noise}')

    # raw_config['dp']['sigma'] = noise

    # dump_config(raw_config, args.config)