import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from eval_tstr import train_tstr


data_path = 'data/adult'
exp_path = 'exp/adult/ctgan/1'
params_path = 'configs/adult/catboost.json'
per_seed = {}

for seed in range(5):
    seed_results = {}

    seed_results = train_tstr(
        data_path,
        exp_path,
        seed=seed,
        eval_type='synthetic',
        params_path=params_path,
        model_names=None,
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
if mode == 'synthetic':
    quality_report = column_quality(data_path, str(sample_dir))

write_json(output / 'results.json', {
    'models': models, 
    'mode': mode,
    'sample_seed': raw_config['sample']['seed'] if mode == 'synthetic' else None,
    **result,
    'quality': quality_report
})
print(f'Results saved to {output / "results.json"}')