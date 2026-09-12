"""Resolve configuration once; paths in version 2 are relative to the TOML file."""
from copy import deepcopy
from pathlib import Path
from tabulargen.io import load_config


def resolve_config(path):
    path = Path(path).resolve()
    raw = load_config(path)
    if raw.get('schema_version') != 2:
        raise ValueError('Expected schema_version = 2; see docs/configuration.md for migration')
    c = deepcopy(raw)
    seed = c.setdefault('seed', 0)
    for section in ['data', 'experiment', 'encoding', 'model', 'diffusion', 'train', 'sample', 'evaluation', 'privacy']:
        c.setdefault(section, {})
    for section in ['encoding', 'train', 'sample', 'evaluation']:
        c[section].setdefault('seed', seed)
        if not isinstance(c[section]['seed'], int) or not 0 <= c[section]['seed'] < 2**32:
            raise ValueError(f'{section}.seed must be an integer in [0, 2**32)')
    c.setdefault('device', 'cpu')
    c['encoding'] = {'num_encoder': 'minmax', 'cat_encoder': 'alb',
                     'histogram_epsilon': 0.1, 'histogram_delta': 1e-5, **c['encoding']}
    c['train'] = {'epochs': 50, 'lr': 3e-4, 'weight_decay': 0.0,
                  'batch_size': 128, 'num_workers': 2, **c['train']}
    c['sample'] = {'num_samples': 512, 'batch_size': 256, 'class_distribution': 'empirical', **c['sample']}
    c['evaluation'] = {'model': 'all', 'mode': 'synthetic', **c['evaluation']}
    c['privacy'] = {'is_dp': False, 'epsilon': 1.0, 'delta': 1e-5, 'max_grad_norm': 1.0, **c['privacy']}
    for section in ['data', 'experiment']:
        if 'path' not in c[section]:
            raise ValueError(f'{section}.path is required')
        c[section]['path'] = str((path.parent / c[section]['path']).resolve())
    if 'catboost_params_path' in c['evaluation']:
        params_path = (path.parent / c['evaluation']['catboost_params_path']).resolve()
        c['evaluation']['catboost_params_path'] = str(params_path)
    if c['evaluation']['model'] not in ['all', 'catboost', 'simple']:
        raise ValueError('evaluation.model must be all, catboost or simple')
    if c['evaluation']['mode'] not in ['real', 'synthetic']:
        raise ValueError('evaluation.mode must be real or synthetic')
    if c['sample']['class_distribution'] not in ['empirical', 'uniform']:
        raise ValueError('sample.class_distribution must be empirical or uniform')
    if c['encoding']['num_encoder'] not in ['minmax', 'standard', 'quantile']:
        raise ValueError('Unknown numerical encoder')
    if c['encoding']['cat_encoder'] not in ['alb', 'oht']:
        raise ValueError('Unknown categorical encoder')
    for section, keys in [('train', ['epochs', 'batch_size']), ('sample', ['num_samples', 'batch_size'])]:
        for key in keys:
            if not isinstance(c[section][key], int) or c[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if not isinstance(c['train']['num_workers'], int) or c['train']['num_workers'] < 0:
        raise ValueError('train.num_workers must be a non-negative integer')
    if c['encoding']['histogram_epsilon'] <= 0 or not 0 < c['encoding']['histogram_delta'] < 1:
        raise ValueError('Encoding histogram requires epsilon > 0 and 0 < delta < 1')
    c['config_source'] = str(path)
    return c
