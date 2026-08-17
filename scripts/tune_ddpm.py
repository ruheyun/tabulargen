import subprocess
import optuna
import shutil
import argparse
from pathlib import Path
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import load_config, dump_config, load_json, dump_json, print_metrics, TabularDataset


def objective(trial):

    base_config = load_config(base_config_path)

    lr = trial.suggest_float('lr', 0.00001, 0.003, log=True)
    base_config['train']['main']['lr'] = lr

    exp_dir = exps_path / f"{trial.number}"
    base_config['exp_path'] = str(exp_dir)
    base_config['eval']['type']['eval_model'] = args.eval_model
    base_config['model_params']['d_in'] = num_features

    trial.set_user_attr("config", base_config)

    dump_config(base_config, exps_path / 'config.toml')
    
    subprocess.run([
        python_exec, f'scripts/{pipeline}',
        '--config', f'{exps_path / "config.toml"}',
        '--train'
    ], check=True)

    n_datasets = 5
    score = 0.0
    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        dump_config(base_config, exps_path / 'config.toml')

        subprocess.run([
            python_exec, f'scripts/{pipeline}',
            '--config', f'{exps_path / "config.toml"}',
            '--sample', '--eval'
        ], check=True)

        report_path = str(Path(base_config['exp_path']) / f'results_{args.eval_model}.json')
        report = load_json(report_path)

        score += report['val']['roc_auc']

    shutil.rmtree(exp_dir, ignore_errors=True)

    return score / n_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default='adult')
parser.add_argument('--cf_name', type=str, default='config')
parser.add_argument('--eval_model', type=str, default='catboost')
parser.add_argument('--prefix', type=str, default='dm')
parser.add_argument('--num_trials', type=int, default=10)

args = parser.parse_args()

python_exec = sys.executable
print(f"[INFO] Using Python executable: {python_exec}")

ds_name = args.ds_name
config_name = args.cf_name
eval_model = args.eval_model
num_trials = args.num_trials
prefix = str(args.prefix + '_' + eval_model)

pipeline = f'pipeline.py'
base_config_path = f'configs/{ds_name}/{config_name}.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = parent_path / 'many-exps'

dataset = TabularDataset(parent_path)
num_features = dataset.X_dim

os.makedirs(exps_path, exist_ok=True)

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=0),
    direction='maximize',
)

func = lambda trial: objective(trial)

study.optimize(func, n_trials=num_trials, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["exp_path"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
dump_config(best_config, best_config_path)

# dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

# subprocess.run([
#     python_exec, f'scripts/{pipeline}',
#     '--config', f'{best_config_path}',
#     '--train'
# ], check=True)

# all_report = []
# for sample_seed in range(5):
#     subprocess.run([
#         python_exec, f'scripts/{pipeline}',
#         '--config', f'{best_config_path}',
#         '--sample', 
#         '--eval',
#         '--sample_seed', f'{sample_seed}'
#     ], check=True)

#     report_path = str(Path(best_config['exp_path']) / f'results_{eval_model}.json')
#     report = load_json(report_path)

#     all_report.append(report)

# avg_results = {
#         split: {
#             metric: round(sum(res[split][metric] for res in all_report) / 5, 4)
#             for metric in ['f1', 'accuracy', 'roc_auc']
#         }
#         for split in ['val', 'test']
#     }

# print_metrics(avg_results)
    