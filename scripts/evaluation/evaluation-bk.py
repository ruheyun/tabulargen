import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import average_metrics, print_metrics
from eval_catboost import train_catboost
from eval_simple import train_simple

def evaluate_models(
    data_path, 
    sample_path, 
    *, 
    model='all', 
    encoded_path=None,
    seed=0, 
    eval_type='synthetic', 
    params=None
):
    params = params or {}
    if model not in ('all', 'catboost', 'simple'):
        raise ValueError(f'Unknown evaluator: {model}')
    if model == 'all':
        unknown = set(params) - {'catboost', 'tree', 'rf', 'lr', 'mlp'}
        if unknown:
            raise ValueError(f'Unknown model parameters: {sorted(unknown)}')
    per_model = {}
    if model in ('catboost', 'all'):
        result = train_catboost(data_path, sample_path, seed=seed, eval_type=eval_type,
                                params=params.get('catboost') if model == 'all' else params)
        per_model.update(result['per_model'])
    if model in ('simple', 'all'):
        simple_params = {name: value for name, value in params.items() if name != 'catboost'}
        result = train_simple(data_path, sample_path, seed=seed, eval_type=eval_type,
                              params=simple_params, encoded_path=encoded_path)
        per_model.update(result['per_model'])
    metrics = average_metrics(per_model)
    if model == 'all':
        print(f'Equal-weight average across {len(per_model)} models: {", ".join(per_model)}')
        print_metrics(metrics)
    return {'metrics': metrics, 'per_model': per_model}
