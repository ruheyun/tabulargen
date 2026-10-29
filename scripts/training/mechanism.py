from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import argparse
import torch
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp, log_ndtr
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
    lst = np.sqrt(lst.reshape(-1, 1))
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit_predict(lst)  

    # 计算分界点
    centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.sort(centers)  
    # print("排序后的簇质心:", sorted_centers)

    # 相邻质心中点作为分界点
    boundaries_noise = (sorted_centers[:-1] + sorted_centers[1:]) / 2

    # 找对应的时间步分界（近似）
    t_boundaries = [0]
    for b in boundaries_noise:
        idx = np.argmin(np.abs(lst.flatten() - b))
        t_boundaries.append(X[idx])
    t_boundaries.append(len(X))
    t_boundaries = np.array(t_boundaries)
    # print("对应的时间步分界点（近似）:", t_boundaries)

    return np.array([(t_boundaries[i], t_boundaries[i+1]) for i in range(K)])[::-1]


def mu_allocation(K, type):
    k = np.arange(1, K + 1)

    if type == 'HA':
        # 1. Harmonic Allocation
        ratio = (1/k - 1/(k + 1)) * (K + 1) / K
    elif type == 'EA':
        # 2. Exponential Allocation
        ratio = (2**(K - k)) / (2**K - 1)
    else:
        raise ValueError('type is error')
    
    mu_ratio = np.sqrt(ratio / ratio.sum())  # ratio.sum() == 1
    return np.sort(mu_ratio)


def epoch_allocation(K, total_epochs, type='equal'):
    k = np.arange(1, K + 1)
    if type == 'decay':
        weight = 1.0 / k
        weight = weight / weight.sum()
        epochs = np.round(weight * total_epochs).astype(int)
        epochs = np.maximum(epochs, 1)
        diff = total_epochs - epochs.sum()
        epochs[0] += diff
        return epochs
    elif type == 'equal':
        base = total_epochs // K
        remainder = total_epochs % K
        epochs = np.full(K, base)
        epochs[:remainder] += 1
        return epochs
    else:
        raise ValueError('type is error')
    

def k_dp_params(K, lst, X, train_iter, epochs, epsilon, delta, account='gdp'):
    t_bd = k_mearns(K, lst, X)

    mu_rt = mu_allocation(K, type='EA')

    epoch_alt = epoch_allocation(K, epochs, type='equal')
    steps_alt = epoch_alt * len(train_iter)

    noise_multiplier = []

    sample_rate = 1 / len(train_iter)
    steps = epochs * len(train_iter)
    accountant = Accountant(sample_rate, steps)
    mu_total = accountant.gdp_get_mu_from_epsilon_delta(epsilon, delta)
    for k in range(K):
        noise_multiplier.append(
            accountant.gdp_get_noise_multiplier_from_mu(
                mu_total * mu_rt[k],
                steps_alt[k]
            )
        )
    return t_bd, noise_multiplier

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

    def gdp_get_mu(self, noise_multiplier, steps=None):
        sigma = noise_multiplier
        steps = self.steps if steps is None else steps

        if sigma <= 0 or steps <= 0 or not 0 < self.q <= 1:
            raise ValueError("Invalid sigma, steps or sample rate")

        # 无子采样：高斯机制的精确 GDP 组合
        if self.q == 1:
            return np.sqrt(steps) / sigma

        # Poisson 子采样：CLT 近似
        return self.q * np.sqrt(
            steps * np.expm1(1.0 / sigma**2)
        )

    @staticmethod
    def gdp_delta_from_mu(epsilon, mu):
        # if mu <= 0:
        #     return 0.0

        # return (
        #     norm.cdf(mu / 2 - epsilon / mu)
        #     - np.exp(epsilon)
        #     * norm.cdf(-mu / 2 - epsilon / mu)
        # )
        if epsilon < 0 or mu < 0:
            raise ValueError("epsilon and mu must be non-negative")

        if mu == 0:
            return 0.0

        a = mu / 2 - epsilon / mu
        b = -mu / 2 - epsilon / mu

        # 在 log 域中计算，避免 exp(epsilon) 溢出
        log_a = log_ndtr(a)
        log_b = epsilon + log_ndtr(b)

        if log_b >= log_a:
            return 0.0

        return float(
            np.exp(log_a) * (-np.expm1(log_b - log_a))
        )

    def gdp_get_epsilon(self, noise_multiplier, delta):
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")

        mu = self.gdp_get_mu(noise_multiplier)

        if delta >= self.gdp_delta_from_mu(0.0, mu):
            return 0.0

        def func(epsilon):
            return self.gdp_delta_from_mu(epsilon, mu) - delta

        lo, hi = 0.0, 1.0
        while func(hi) > 0:
            hi *= 2.0

        return brentq(func, lo, hi)

    def gdp_get_noise_multiplier(self, epsilon, delta):
        mu = self.gdp_get_mu_from_epsilon_delta(
            epsilon, delta
        )
        return self.gdp_get_noise_multiplier_from_mu(mu)

    def gdp_get_mu_from_epsilon_delta(self, epsilon, delta):
        if epsilon < 0 or not 0 < delta < 1:
            raise ValueError("Invalid epsilon or delta")

        def func(mu):
            return self.gdp_delta_from_mu(epsilon, mu) - delta

        lo, hi = 0.0, 1.0
        while func(hi) < 0:
            hi *= 2.0

        return brentq(func, lo, hi)

    def gdp_get_noise_multiplier_from_mu(self, mu, steps=None):
        if mu <= 0:
            raise ValueError("mu must be positive")

        steps = self.steps if steps is None else steps

        if steps <= 0 or not 0 < self.q <= 1:
            raise ValueError("Invalid steps or sample rate")

        if self.q == 1:
            return np.sqrt(steps) / mu

        # 解析反解 sigma
        return 1.0 / np.sqrt(
            np.log1p((mu / self.q)**2 / steps)
        )
