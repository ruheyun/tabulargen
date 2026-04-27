import torch
import os
import json
import tomllib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from inspect import isfunction
from torch.utils.data import Dataset
from typing import Union, Any, Dict, cast
from pathlib import Path

RawConfig = Dict[str, Any]
_CONFIG_NONE = '__none__'


def load_config(path: Union[Path, str]) -> Any:
    """
    读取 TOML 配置文件。
    """
    path = Path(path)
    with path.open("rb") as f:
        config = tomllib.load(f)
    return unpack_config(config)


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    t = t.to(a.device)
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def exists(x):
    return x is not None


def update_ema(target_params, source_params, rate=0.999):

    with torch.no_grad():
        for targ, src in zip(target_params, source_params):
            targ.mul_(rate).add_(src.detach(), alpha=1 - rate)


class TabularDataset(Dataset):

    def __init__(self, data_path, type='train'):

        df = pd.read_csv(os.path.join(data_path, f'{type}.csv'))

        with open(os.path.join(data_path, 'info.json'), 'r') as f:
            info = json.load(f)     

        if info['task_type'] == 'binclass':
            label_dtype = torch.float32
        else:
            label_dtype = torch.long
        
        self.y = torch.tensor(df['label'].values, dtype=label_dtype)
        self.X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
        
        self.X_dim = self.X.shape[1]

        assert self.X_dim == info['n_features'], ('data dim false!')

    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
    