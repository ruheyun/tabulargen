import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]


def timestep_embedding(timesteps, dim, max_period=10000):
    
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def reglu(x: Tensor) -> Tensor:
    
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)
    

class GEGLU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)
    

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):

    class Block(nn.Module):

        def __init__(
                self,
                *,
                d_in: int,
                d_out: int,
                bias: bool,
                activation: ModuleType,
                dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
            self,
            *,
            d_in: int,
            d_layers: List[int],
            dropouts: Union[float, List[float]],
            activation: Union[str, Callable[[], nn.Module]],
            d_out: int,
    ) -> None:
        
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
            cls: Type['MLP'],
            d_in: int,
            d_layers: List[int],
            dropout: float,
            d_out: int,
    ) -> 'MLP':
        
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, num_classes, is_y_cond, rtdl_params, dim_t=128):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        rtdl_params['d_in'] = dim_t
        rtdl_params['d_out'] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)

        if self.num_classes > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb = emb + F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)
    