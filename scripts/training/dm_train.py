from copy import deepcopy
import csv
import json
import torch
import numpy as np
import delu
from tqdm import trange, tqdm
import pandas as pd
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models import GaussianDiffusion, MLPDiffusion
from utils import update_ema, TabularDataset
from mechanism import Accountant


class Trainer:
    def __init__(self, diffusion, ema_model, train_iter, lr, optimizer, dp_params,
                epochs, info, loss_history, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.ema_model = ema_model
        self.train_iter = train_iter
        self.dp_params = dp_params
        self.init_lr = lr
        self.optimizer = optimizer
        self.device = device
        self.loss_history = loss_history
        self.log_every = 10
        self.epochs = epochs
        self.steps = epochs * len(train_iter)
        self.info = info
        self.is_dp = dp_params['is_dp']
        self.epsilon = dp_params['epsilon']
        self.delta = dp_params['delta']
        self.max_grad_norm = dp_params['max_grad_norm']

        if self.is_dp:
            sample_rate = 1 / len(train_iter)
            accountant = Accountant(sample_rate, self.steps)
            gdp_noise = accountant.gdp_get_noise_multiplier(epsilon=self.epsilon, delta=self.delta)
            # rdp_noise = accountant.rdp_get_noise_multiplier(epsilon=self.epsilon, delta=self.delta)
            # ma_noise = accountant.ma_get_noise_multiplier(epsilon=self.epsilon, delta=self.delta)
            noise_multiplier = gdp_noise

            print(f'noise: {noise_multiplier}')

            self.privacy_engine = PrivacyEngine()
            self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private(
                module=self.diffusion,
                optimizer=self.optimizer,
                data_loader=self.train_iter,
                max_grad_norm=self.max_grad_norm,
                noise_multiplier=noise_multiplier
            )
            self.diffusion.compute_loss = self.diffusion._module.compute_loss
    
    def _anneal_C(self, step):
        C = 0.5 + (2 - 0.5) * np.exp(-5 * step / self.steps)
        self.optimizer.max_grad_norm = C

    def _anneal_lr(self, step):
        frac_done = min(0.999999, step / self.steps)
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.compute_loss(x, out_dict)
        loss.backward()

        self.optimizer.step()

        return loss

    def run_loop(self):
        step = 0
        pbar = tqdm(iterable=range(self.epochs), position=0, leave=True)
        for epoch in range(self.epochs):
            curr_loss_gauss = 0.0
            curr_count = 0
            for x, out_dict in self.train_iter:
                out_dict = {'y': out_dict}
                batch_loss_gauss = self._run_step(x, out_dict)

                curr_count += len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                self._anneal_lr(step)
                step += 1

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                if (step + 1) % self.log_every == 0:
                    loss = np.around(curr_loss_gauss / curr_count, 3)
                    self.loss_history.loc[len(self.loss_history)] = [step + 1, loss]
            
            # pbar.set_postfix({'Loss': round(loss, 3),})
            loss = np.around(curr_loss_gauss / curr_count, 3)
            pbar.set_description(f"Epoch {epoch + 1:04d} | Train Loss: {loss:.3f}")
            pbar.update(1)
             
        print(
            f'({self.epsilon}, {self.delta})-DP training done!'
            if self.is_dp else 'No-DP training done!'
        )


def train(
        exp_path='exp/adult/run_00/encoded',
        epochs=50,
        lr=1e-4,
        weight_decay=0.0,
        batch_size=128,
        model_params=None,
        num_timesteps=500,
        gaussian_loss_type='mse',
        scheduler='cosine',
        dp_params=None,
        device=torch.device('cuda:0'),
        seed=0,
        checkpoint_path=None,
        log_path=None,
        num_workers=2,
):
    delu.random.seed(seed)

    device = torch.device(device)

    with open(os.path.join(exp_path, 'info.json'), 'r') as f:
        info = json.load(f)

    dataset = TabularDataset(exp_path)

    num_features = dataset.X_dim
    model_params = deepcopy(model_params)
    model_params['d_in'] = num_features

    print(f'model params: {model_params}\ndevice: {device}')

    loss_history = pd.DataFrame(columns=['step', 'loss'])

    model = MLPDiffusion(**model_params)
    model.to(device)

    diffusion = GaussianDiffusion(
        input_dim=num_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        dp_params=dp_params,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    ema_model = deepcopy(diffusion._denoise_fn)
    for param in ema_model.parameters():
        param.detach_()

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=device.type == 'cuda')

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(
        diffusion,
        ema_model,
        train_loader,
        lr,
        optimizer,
        dp_params,
        epochs,
        info,
        loss_history,
        device
    )
    trainer.run_loop()

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(checkpoint_path, 'model.pt'))
    # torch.save(ema_model.state_dict(), os.path.join(checkpoint_path, 'model_ema.pt'))

    torch.save(
        {
            'model_state_dict': diffusion._denoise_fn.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'model_params': model_params,
            'diffusion_params': {
                'num_timesteps': num_timesteps,
                'gaussian_loss_type': gaussian_loss_type,
                'scheduler': scheduler
            },
            'encoding': {
                'path': os.path.relpath(Path(exp_path).resolve(), Path(checkpoint_path).resolve())
            },
            'train_params': {
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'seed': seed,
                'num_workers': num_workers,
                'dp_params': deepcopy(dp_params)
            }
        },
        os.path.join(checkpoint_path, 'checkpoint.pt')
    )

    loss_history.to_csv(os.path.join(log_path, 'loss.csv'), index=False)    
