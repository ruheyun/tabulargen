import json
import pickle
import pandas as pd
import torch
import numpy as np
import delu
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models import GaussianDiffusion, MLPDiffusion


def sample(
    exp_path='exp/adult',
    batch_size=256,
    num_samples=0,
    model_params=None,
    model_path=None,
    num_timesteps=500,
    gaussian_loss_type='mse',
    scheduler='cosine',
    device=torch.device('cuda:0'),
    seed=0,
):
    delu.random.seed(seed)

    with open(os.path.join(exp_path, 'info.json'), 'r') as f:
        info = json.load(f)  

    model = MLPDiffusion(**model_params)

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    diffusion = GaussianDiffusion(
        input_dim=info['n_features'],
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )

    diffusion.to(device)
    diffusion.eval()

    print('Starting sampling...')
    empirical_class_dist = torch.tensor(info['p_y'], dtype=torch.float32)
    
    x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist)

    X_gen, y_gen = x_gen.cpu().numpy(), y_gen.cpu().numpy()

    with open(f"{exp_path}/data_wrapper.pkl", "rb") as f:
        data_wrapper = pickle.load(f)

    with open(f"{exp_path}/label_wrapper.pkl", "rb") as f:
        label_wrapper = pickle.load(f)

    X_gen_ = data_wrapper.Reverse(X_gen)
    y_gen_ = label_wrapper.Reverse(y_gen)

    X_gen = pd.DataFrame(X_gen)
    y_gen = pd.DataFrame(y_gen)

    num_cols = [f"num_{i}" for i in range(info['n_num_features'])]
    cat_cols = [f"cat_{i}" for i in range(info['n_features'] - info['n_num_features'])]
    y_cols = ['label']

    cols = num_cols + cat_cols + y_cols

    unreverse_data = pd.concat([X_gen, y_gen], axis=1)
    unreverse_data.columns = cols
    unreverse_data.to_csv(os.path.join(exp_path, 'unreverse.csv'), index=False, header=True)

    X_gen_ = pd.DataFrame(X_gen_)
    y_gen_ = pd.DataFrame(y_gen_)

    y_gen_.columns = info['y_name']

    reverse_data = pd.concat([X_gen_, y_gen_], axis=1)
    reverse_data.to_csv(os.path.join(exp_path, 'reverse.csv'), index=False, header=True)

    print(f"Raw samples saved to {exp_path}, Sample done!")
