import json
import pandas as pd
import torch
import delu
import os
from copy import deepcopy
from tabulargen.models import GaussianDiffusion, MLPDiffusion
from tabulargen.data.encoding import load_wrapper
from tabulargen.checkpoints import load_checkpoint


def sample(
    exp_path,
    model_path,
    batch_size=256,
    num_samples=0,
    device=torch.device('cuda:0'),
    seed=0,
    disbalance=None
):
    delu.random.seed(seed)

    if num_samples <= 0 or batch_size <= 0:
        raise ValueError('num_samples and batch_size must be positive')
    checkpoint, encoding_path = load_checkpoint(model_path)
    model_params = deepcopy(checkpoint['model_params'])
    diffusion_params = checkpoint['diffusion_params']
    state_dict = checkpoint['ema_state_dict']

    with open(encoding_path / 'info.json', 'r') as f:
        info = json.load(f)

    if model_params['d_in'] != info['encoded_dim']:
        raise ValueError('Model input dimension does not match encoded data')
    model = MLPDiffusion(**model_params)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    diffusion = GaussianDiffusion(
        input_dim=info['encoded_dim'],
        denoise_fn=model,
        **diffusion_params,
        device=device
    )

    diffusion.to(device)
    diffusion.eval()

    print('Starting sampling...')
    if disbalance == 'uniform':
        empirical_class_dist = torch.ones(info['n_classes'], dtype=torch.float32) / info['n_classes']
    else:
        empirical_class_dist = torch.tensor(info['origin_p_y'], dtype=torch.float32)

    x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist)

    X_gen, y_gen = x_gen.cpu().numpy(), y_gen.cpu().numpy()

    data_wrapper = load_wrapper(encoding_path / 'data_wrapper.pkl')

    label_wrapper = load_wrapper(encoding_path / 'label_wrapper.pkl')

    X_gen_ = data_wrapper.Reverse(X_gen)
    y_gen_ = label_wrapper.Reverse(y_gen)

    X_gen = pd.DataFrame(X_gen)
    y_gen = pd.DataFrame(y_gen)

    num_cols = [f"num_{i}" for i in range(info['n_num_features'])]
    cat_cols = [f"cat_{i}" for i in range(info['encoded_dim'] - info['n_num_features'])]
    y_cols = ['label']

    cols = num_cols + cat_cols + y_cols

    unreverse_data = pd.concat([X_gen, y_gen], axis=1)
    unreverse_data.columns = cols
    os.makedirs(exp_path, exist_ok=True)
    unreverse_data.to_csv(os.path.join(exp_path, 'unreverse.csv'), index=False, header=True)

    X_gen_ = pd.DataFrame(X_gen_)
    y_gen_ = pd.DataFrame(y_gen_)

    y_gen_.columns = info['y_name']

    reverse_data = pd.concat([X_gen_, y_gen_], axis=1)
    reverse_data.to_csv(os.path.join(exp_path, 'reverse.csv'), index=False, header=True)

    print(f"Raw samples saved to {exp_path}, Sample done!")
    return encoding_path
