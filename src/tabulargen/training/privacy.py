from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import argparse
import torch
import math
import os
import sys
from tabulargen.data.dataset import TabularDataset
from tabulargen.io import load_config, dump_config


def dp_histogram(labels, num_classes, epsilon=0.1, delta=1e-5):

    labels = torch.tensor(labels.values, dtype=torch.long).view(-1)

    counts = torch.bincount(labels, minlength=num_classes).float()

    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    noise = torch.normal(0, sigma, size=counts.shape)
    noisy_counts = counts + noise

    noisy_counts = torch.clamp(noisy_counts, min=1e-6)

    p_y = noisy_counts / noisy_counts.sum()

    return p_y.numpy()


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
    return noise_multiplier
