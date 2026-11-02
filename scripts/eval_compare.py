import os
import sys
from pathlib import Path
import json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from scripts.evaluation.eval_tstr import train_tstr
from utils import load_config, RunPaths, average_metrics, print_metrics
from scripts.evaluation.eval_metrics import column_quality


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

dns = ['adult', 'bank', 'beijing', 'buddy', 'california', 'cardio', 'default', 'house', 'king', 'market', 'shoppers']
mdls = ['ctgan', 'tvae', 'findiff']
eps = ['1', '5', '10']

models = ['catboost', 'tree', 'rf', 'lr', 'mlp']
for dn in dns:
    for mdl in mdls:
        if dn == 'california' and mdl == 'findiff':
            continue
        if dn == 'house' and mdl == 'findiff':
            continue
        for ep in eps:
            data_path = f'data/{dn}'
            exp_path = f'exp/{dn}/{mdl}/{ep}'
            params_path = f'configs/{dn}/catboost.json'

            per_seed = {}

            for seed in range(5):
                seed_results = {}

                seed_results = train_tstr(
                    data_path,
                    exp_path,
                    seed=seed,
                    eval_type='synthetic',
                    params_path=params_path,
                    model_names=models,
                )

                per_seed[seed] = seed_results['per_model']

            per_model = {
                name: average_metrics({
                    seed: results[name]
                    for seed, results in per_seed.items()
                })
                for name in models
            }

            groups = {}
            if 'catboost' in per_model:
                groups['catboost'] = per_model['catboost']

            simple_results = {
                name: result
                for name, result in per_model.items()
                if name != 'catboost'
            }
            if simple_results:
                groups['simple'] = average_metrics(simple_results)

            metrics = average_metrics(groups)
            print('Average results')
            print_metrics(metrics)

            result = {'metrics': metrics, 'per_model': per_model}

            quality_report = None
            quality_report = column_quality(data_path, exp_path)

            write_json(os.path.join(exp_path, 'results.json'), {
                'models': models, 
                **result,
                'quality': quality_report
            })
            print(f'Results saved to {exp_path}/results.json')
