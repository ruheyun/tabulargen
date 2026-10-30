from copy import deepcopy
import json
import torch
import numpy as np
import delu
from tqdm import trange, tqdm
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models import GaussianDiffusion, MLPDiffusion
from utils import update_ema, TabularDataset
from .mechanism import k_dp_params


class Trainer:
    def __init__(self, diffusion, ema_model, train_iter, lr, optimizer, dp_params,
                epochs, info, loss_history, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.num_timesteps = diffusion.num_timesteps
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
        self.step = 0
        self.info = info
        self.is_dp = dp_params['is_dp']
        self.epsilon = dp_params['epsilon']
        self.delta = dp_params['delta']
        self.max_grad_norm = dp_params['max_grad_norm']
        self.K = dp_params['K']
    
    def _anneal_C(self, step):
        C = 0.5 + (2 - 0.5) * np.exp(-5 * step / self.steps)
        self.optimizer.max_grad_norm = C

    def _anneal_lr(self, step):
        frac_done = min(0.99, step / self.steps)
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict, ts=None):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.compute_loss(x, out_dict, ts)
        loss.backward()

        self.optimizer.step()

        return loss
        
    def run_loop(self, epoch_alt=None, ts=None):
        epochs = self.epochs if epoch_alt is None else epoch_alt
        pbar = tqdm(iterable=range(epochs), position=0, leave=True)
        for epoch in range(epochs):
            curr_loss_gauss = 0.0
            curr_count = 0
            for x, out_dict in self.train_iter:
                if self.info['task_type'] == 'regression':
                    out_dict = {}
                else:
                    out_dict = {'y': out_dict}

                batch_loss_gauss = self._run_step(x, out_dict, ts)

                curr_count += len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                # self._anneal_lr(self.step)
                self.scheduler.step()
                self.step += 1

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                if (self.step + 1) % self.log_every == 0:
                    loss = np.around(curr_loss_gauss / curr_count, 3)
                    self.loss_history.loc[len(self.loss_history)] = [self.step + 1, loss]
            
            # pbar.set_postfix({'Loss': round(loss, 3),})
            loss = np.around(curr_loss_gauss / curr_count, 3)
            pbar.set_description(f"Epoch {epoch + 1:04d} | Train Loss: {loss:.3f}")
            pbar.update(1)
             
        print(
            f'({self.epsilon}, {self.delta})-DP training done!'
            if self.is_dp else 'No-DP training done!'
        )

    def run(self):
        noise_multiplier = None
        if self.is_dp:
            t_bd, epochs_alt, noise_multiplier = k_dp_params(
                self.K,
                self.diffusion.noise_intensity,
                np.arange(self.num_timesteps),
                self.train_iter,
                self.epochs,
                self.epsilon,
                self.delta,
            )

            print(f'noise: {noise_multiplier}')

            self.privacy_engine = PrivacyEngine()
            self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private(
                module=self.diffusion,
                optimizer=self.optimizer,
                data_loader=self.train_iter,
                max_grad_norm=self.max_grad_norm,
                noise_multiplier=noise_multiplier[0]
            )
            self.diffusion.compute_loss = self.diffusion._module.compute_loss

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.steps,
            eta_min=self.init_lr * 0.01,
        )

        if self.is_dp:
            for k in range(self.K):
                self.optimizer.noise_multiplier = noise_multiplier[k]
                self.run_loop(epoch_alt=epochs_alt[k], ts=t_bd[k])
        else:
            self.run_loop()
    

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
        ratio=1,
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
        ratio=ratio,
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
    trainer.run()

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    torch.save(
        {
            'model_state_dict': diffusion._denoise_fn.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'model_params': model_params,
            'diffusion_params': {
                'num_timesteps': num_timesteps,
                'gaussian_loss_type': gaussian_loss_type,
                'scheduler': scheduler,
                'ratio': ratio
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
