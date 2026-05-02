import os
import optuna
import argparse
import json
from eval_catboost import train_catboost


def suggest_catboost_params(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
        "depth": trial.suggest_int("depth", 3, 10), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10)
    }

    params = params | {
        "iterations": 3000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU",
        "thread_count": 4,
    }
    return params


def objective(trial):
    
    params = suggest_catboost_params(trial)
    
    trial.set_user_attr("params", params)

    results = train_catboost(
        data_path=data_path,
        exp_path=exp_path,
        eval_type="real",
        params=params,
    )
    score = results['val']['f1']

    return score


parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default='bank')
parser.add_argument('--n_trials', type=int, default=100)

args = parser.parse_args()
data_name = args.ds_name
n_trials = args.n_trials
data_path = os.path.join('data', data_name)
exp_path = os.path.join('exp', data_name)

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=0),
    direction='maximize'
)

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

best_params = study.best_trial.user_attrs['params']

with open(os.path.join('configs', data_name, 'catboost.json'), 'w') as f:
    json.dump(best_params, f)
