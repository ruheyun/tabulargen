import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARCHITECTURES = {
    'small': [128, 128],
    'medium': [256, 256],
    'current': [128, 256, 256, 128],
    'large': [256, 512, 512, 256],
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


def objective(trial, base_config, exps_path, eval_model, num_seeds):
    from utils import dump_config, load_json

    config = deepcopy(base_config)
    config['train']['lr'] = trial.suggest_float('lr', 1e-5, 0.003, log=True)
    config['train']['batch_size'] = trial.suggest_categorical('batch_size', [128, 256, 512])
    architecture = trial.suggest_categorical('architecture', list(ARCHITECTURES))
    config['model']['rtdl_params']['d_layers'] = ARCHITECTURES[architecture].copy()
    config['diffusion']['num_timesteps'] = trial.suggest_categorical('num_timesteps', [500, 1000])
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
        / f"seed_{config['evaluation']['seed']}" / 'results.json'
    )
    for sample_seed in range(num_seeds):
        run_pipeline(config_path, '--sample', '--eval', '--sample_seed', str(sample_seed))
        report = load_json(report_path)
        if report['sample_seed'] != sample_seed:
            raise ValueError(f'Unexpected sample seed in {report_path}')
        score = float(report['per_model'][eval_model]['val']['roc_auc'])
        if not math.isfinite(score):
            raise ValueError(f'Non-finite validation ROC-AUC in {report_path}')
        scores.append(score)
        # The pipeline reuses the evaluation-seed directory for every sample seed.
        (exp_dir / f'results_sample_seed_{sample_seed}.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n'
        )
        trial.set_user_attr('sample_scores', scores.copy())

    mean_score = sum(scores) / len(scores)
    (exp_dir / 'scores.json').write_text(json.dumps({
        'eval_model': eval_model,
        'sample_seeds': list(range(num_seeds)),
        'val_roc_auc': scores,
        'mean_val_roc_auc': mean_score,
    }, indent=2, allow_nan=False) + '\n')
    return mean_score


def main():
    parser = argparse.ArgumentParser(
        description='Tune DDPM learning rate, training batch size, architecture and diffusion timesteps.'
    )
    parser.add_argument('--config', type=Path, required=True,
                        help='Base config; relative paths are resolved from the project root.')
    parser.add_argument('--ds_name', default=None, help='Optional check against data.path.')
    parser.add_argument('--num_trials', type=positive_int, default=30,
                        help='Number of trials (default: 30).')
    parser.add_argument('--num_seeds', type=positive_int, default=3,
                        help='Number of shared sampling seeds per trial (default: 3).')
    parser.add_argument('--eval_model', choices=['catboost', 'tree', 'rf', 'lr', 'mlp'],
                        default='catboost')
    args = parser.parse_args()

    import optuna
    from utils import load_config, dump_config

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    base_config = load_config(config_path)
    ds_name = validate_config(base_config, args.ds_name)
    base_config['evaluation']['models'] = [args.eval_model]
    base_config['evaluation']['mode'] = 'synthetic'
    if args.eval_model == 'catboost':
        params_path = base_config['evaluation'].get('catboost_params_path')
        if not params_path or not (ROOT / params_path).is_file():
            raise ValueError('evaluation.catboost_params_path must point to an existing file')

    parent_path = ROOT / 'exp' / ds_name / 'many-exps'
    parent_path.mkdir(parents=True, exist_ok=True)
    exps_path = Path(tempfile.mkdtemp(prefix='tune_', dir=parent_path))
    print(f'[INFO] Python: {sys.executable}')
    print(f'[INFO] Config: {config_path}; DP: {base_config["privacy"]["is_dp"]}')
    print(f'[INFO] Tuning output: {exps_path}')

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0), direction='maximize',
    )
    study.optimize(
        lambda trial: objective(trial, base_config, exps_path, args.eval_model, args.num_seeds),
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
        'mean_val_roc_auc': study.best_value,
        'sample_scores': study.best_trial.user_attrs['sample_scores'],
    }, indent=2, allow_nan=False) + '\n')
    print(f'[INFO] Best validation ROC-AUC: {study.best_value:.6f}')
    print(f'[INFO] Best config: {best_config_path}')
    print('[INFO] To train this config, run scripts/pipeline.py --config <best config> --encode --train')


if __name__ == '__main__':
    main()
