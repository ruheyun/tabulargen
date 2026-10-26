import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import load_config, dump_config, load_json


SEARCH_SPACE = {
    'num_timesteps': [500, 1000],
    'scheduler': ['linear', 'cosine'],
    'epochs': [300, 500]
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def validate_config(config, ds_name):
    required = {
        'data': ('path',),
        'experiment': ('path',),
        'encoding': ('num_encoder', 'cat_encoder', 'seed'),
        'model': ('rtdl_params',),
        'diffusion': ('num_timesteps', 'gaussian_loss_type', 'scheduler'),
        'train': ('lr', 'epochs', 'batch_size', 'seed'),
        'sample': ('seed', 'batch_size', 'num_samples', 'class_distribution'),
        'evaluation': ('seed',),
        'privacy': ('is_dp', 'epsilon', 'delta', 'max_grad_norm'),
    }
    missing = []
    for section, keys in required.items():
        values = config.get(section)
        for key in keys:
            if not isinstance(values, dict) or key not in values:
                missing.append(f'{section}.{key}')
    if 'device' not in config:
        missing.append('device')
    if missing:
        raise ValueError('Expected a version-2 config; missing: ' + ', '.join(missing))
    actual_name = Path(config['data']['path']).name
    if ds_name is not None and ds_name != actual_name:
        raise ValueError(f'--ds_name={ds_name} conflicts with data.path={config["data"]["path"]}')
    return actual_name


def run_pipeline(config_path, *flags):
    subprocess.run(
        [sys.executable, str(ROOT / 'scripts/pipeline.py'),
         '--config', str(config_path), *flags],
        cwd=ROOT,
        check=True,
    )


def objective(trial, base_config, exps_path, eval_model, num_seeds, metric):

    config = deepcopy(base_config)
    config['diffusion']['num_timesteps'] = trial.suggest_categorical('num_timesteps', SEARCH_SPACE['num_timesteps'])
    config['diffusion']['scheduler'] = trial.suggest_categorical('scheduler', SEARCH_SPACE['scheduler'])
    config['train']['epochs'] = trial.suggest_categorical('epochs', SEARCH_SPACE['epochs'])
    exp_dir = exps_path / str(trial.number)
    exp_dir.mkdir()
    config['experiment']['path'] = str(exp_dir)
    config_path = exp_dir / 'config.toml'
    dump_config(config, config_path)
    trial.set_user_attr('config', deepcopy(config))

    run_pipeline(config_path, '--encode', '--train')

    scores = []
    report_path = (
        exp_dir / 'evaluation'
        / f"seed_5" / 'results.json'
    )
    for sample_seed in range(num_seeds):
        run_pipeline(config_path, '--sample', '--eval5', '--sample_seed', str(sample_seed))
        report = load_json(report_path)
        if report['sample_seed'] != sample_seed:
            raise ValueError(f'Unexpected sample seed in {report_path}')
        score = float(report['metrics']['val'][metric])
        if not math.isfinite(score):
            raise ValueError(f'Non-finite validation {metric} in {report_path}')
        scores.append(score)
        (exp_dir / f'results_sample_seed_{sample_seed}.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n'
        )
        trial.set_user_attr('sample_scores', scores.copy())

    mean_score = sum(scores) / len(scores)
    (exp_dir / 'scores.json').write_text(json.dumps({
        'eval_model': eval_model,
        'metric': metric,
        'sample_seeds': list(range(num_seeds)),
        'val_scores': scores,
        'mean_val_score': mean_score,
    }, indent=2, allow_nan=False) + '\n')
    return mean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ds_name', default=None)
    parser.add_argument('--num_trials', type=positive_int, default=8)
    parser.add_argument('--num_seeds', type=positive_int, default=3)
    args = parser.parse_args()

    
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    base_config = load_config(config_path)
    base_config['evaluation']['mode'] = 'synthetic'

    info = load_json(ROOT / base_config['data']['path'] / 'info.json')
    metric = (
        'r2'
        if info['task_type'] == 'regression'
        else 'roc_auc'
    )
    ds_name = validate_config(base_config, args.ds_name)

    parent_path = ROOT / 'exp' / ds_name / 'many-exps'
    parent_path.mkdir(parents=True, exist_ok=True)
    exps_path = Path(tempfile.mkdtemp(prefix='tune_', dir=parent_path))
    print(f'[INFO] Python: {sys.executable}')
    print(f'[INFO] Config: {config_path}; DP: {base_config["privacy"]["is_dp"]}')
    print(f'[INFO] Tuning output: {exps_path}')

    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(SEARCH_SPACE, seed=0), 
        direction='maximize',
    )
    study.optimize(
        lambda trial: objective(trial, base_config, exps_path, args.eval_model, args.num_seeds, metric),
        n_trials=args.num_trials, show_progress_bar=True,
    )

    best_dir = exps_path / 'best'
    best_dir.mkdir()
    best_config = deepcopy(study.best_trial.user_attrs['config'])
    best_config['experiment']['path'] = str(best_dir)
    best_config_path = best_dir / 'config.toml'
    dump_config(best_config, best_config_path)
    (best_dir / 'summary.json').write_text(json.dumps({
        'trial': study.best_trial.number,
        'params': study.best_trial.params,
        'metric': metric,
        'mean_val_score': study.best_value,
        'sample_scores': study.best_trial.user_attrs['sample_scores'],
    }, indent=2, allow_nan=False) + '\n')
    print(f'[INFO] Best validation {metric}: {study.best_value:.6f}')
    print(f'[INFO] Best config: {best_config_path}')
    print('[INFO] To train this config, run scripts/pipeline.py --config <best config> --encode --train')


if __name__ == '__main__':
    main()
