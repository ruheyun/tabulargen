import torch
import numpy as np
import csv

from opacus.grad_sample import GradSampleModule


class GradNormAnalyzer:

    def __init__(self, model, csv_path="exp/grad_stats.csv"):

        self.model = GradSampleModule(model)
        self.csv_path = csv_path

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "C95", "mean", "std", "max", "min"
            ])

    def compute_per_sample_norms(self):

        batch_size = None
        per_sample_norms = None

        for p in self.model.parameters():
            if p.grad_sample is None:
                continue

            if batch_size is None:
                batch_size = p.grad_sample.shape[0]
                per_sample_norms = [0.0 for _ in range(batch_size)]

            for i in range(batch_size):
                g_i = p.grad_sample[i]
                per_sample_norms[i] += g_i.pow(2).sum().item()

        per_sample_norms = [np.sqrt(x) for x in per_sample_norms]

        return per_sample_norms

    def log_stats(self):
        norms = self.compute_per_sample_norms()

        norms_np = np.array(norms)

        C95 = np.percentile(norms_np, 95)
        mean = np.mean(norms_np)
        std = np.std(norms_np)
        maxv = np.max(norms_np)
        minv = np.min(norms_np)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([C95, mean, std, maxv, minv])
    