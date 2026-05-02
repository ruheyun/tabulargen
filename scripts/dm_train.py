from copy import deepcopy
import csv
import json
import torch
import numpy as np
import delu
from tqdm import trange
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models import GaussianDiffusion, MLPDiffusion
from utils import update_ema, TabularDataset
from analyze_grad import GradNormAnalyzer


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
        self.max_grad_norm = dp_params['max_grad_norm']
        self.sigma = dp_params['sigma']

        if self.is_dp:
            self.privacy_engine = PrivacyEngine()
            self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private(
                module=self.diffusion,
                optimizer=self.optimizer,
                data_loader=self.train_iter,
                max_grad_norm=self.max_grad_norm,
                noise_multiplier=self.sigma
            )
            self.diffusion.compute_loss = self.diffusion._module.compute_loss
        else:
            self.analyzer = GradNormAnalyzer(self.diffusion)
            self.diffusion = self.analyzer.model
            self.diffusion.compute_loss = self.diffusion._module.compute_loss
    
    def _anneal_C(self, step):
        C = 0.5 + (2 - 0.5) * np.exp(-5 * step / self.steps)
        self.optimizer.max_grad_norm = C

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        if lr < self.init_lr * 0.3:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _gradient_rescaling(self, y, alpha=-0.5, tau=0.01, w_max=5):
        p_y = self.info['p_y']
        p_y = torch.tensor(p_y, dtype=torch.float32, device=self.device)
        p_y_smooth = (p_y + tau) / (1 + tau * len(p_y))
        w_y = p_y_smooth ** alpha
        w_y = w_y / w_y.mean()
        w_y = torch.clamp(w_y, max=w_max)

        for param in self.diffusion._denoise_fn.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                w_expanded = w_y[y]
                while w_expanded.dim() < param.grad_sample.dim():
                    w_expanded = w_expanded.unsqueeze(-1)
                param.grad_sample *= w_expanded

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.diffusion.compute_loss(x, out_dict, is_dp=self.is_dp)
        loss.backward()

        # self.analyzer.log_stats()
        self._gradient_rescaling(out_dict['y'])

        self.optimizer.step()

        # self.analyzer.clear_grad_sample()
        return loss

    def run_loop(self):
        curr_loss_gauss = 0.0
        curr_count = 0
        with trange(self.steps, unit="step", dynamic_ncols=True) as pbar:
            step = 0
            for epoch in range(self.epochs):
                for x, out_dict in self.train_iter:
                    out_dict = {'y': out_dict}
                    batch_loss_gauss = self._run_step(x, out_dict)

                    curr_count += len(x)
                    curr_loss_gauss += batch_loss_gauss.item() * len(x)

                    self._anneal_lr(step)
                    step += 1
                    # self._anneal_C(step)

                    update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                    if (step + 1) % self.log_every == 0:
                        loss = np.around(curr_loss_gauss / curr_count, 3)
                        pbar.set_postfix({
                            'Loss': round(loss, 3),
                        })
                        
                        self.loss_history.loc[len(self.loss_history)] = [step + 1, loss]
                        curr_count = 0
                        curr_loss_gauss = 0.0

                    pbar.update(1)
             
        print(
            f'({self.privacy_engine.get_epsilon(self.dp_params['delta'])}, {self.dp_params['delta']})-DP training done!'
            if self.is_dp else 'No-DP training done!'
        )


def train(
        exp_path='exp/adult',
        epochs=100,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=256,
        model_params=None,
        num_timesteps=500,
        gaussian_loss_type='mse',
        scheduler='cosine',
        dp_params=None,
        device=torch.device('cuda:0'),
        seed=0,
):
    delu.random.seed(seed)

    with open(os.path.join(exp_path, 'info.json'), 'r') as f:
        info = json.load(f)

    dataset = TabularDataset(exp_path)

    num_features = dataset.X_dim
    model_params['d_in'] = num_features

    print(f'model params: {model_params}')

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

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)

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

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(exp_path, 'model.pt'))
    torch.save(ema_model.state_dict(), os.path.join(exp_path, 'model_ema.pt'))

    loss_history.to_csv(os.path.join(exp_path, 'loss.csv'), index=False)    
