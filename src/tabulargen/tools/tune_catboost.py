import argparse
from copy import deepcopy
from pathlib import Path
import optuna
from tabulargen.cli import run, write_json
from tabulargen.config import resolve_config
from tabulargen.layout import RunPaths
import json


def main():
    parser = argparse.ArgumentParser(description='Tune CatBoost on real data without overwriting source configs')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-trials', type=int, default=100)
    args = parser.parse_args()
    config = resolve_config(args.config)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new tuning output directory')
    output.mkdir(parents=True)

    def objective(trial):
        c = deepcopy(config)
        c['experiment']['path'] = str(output / f'trial_{trial.number}')
        c['evaluation'].update(model='catboost', mode='real')
        c['evaluation']['params'] = c['evaluation'].get('params', {}) | {
            'learning_rate': trial.suggest_float('learning_rate', .001, 1., log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', .1, 10.),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0., 1.),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
        }
        trial.set_user_attr('params', c['evaluation']['params'])
        run(c, ['eval'])
        path = RunPaths(Path(c['experiment']['path'])).evaluation('real', 0, 'catboost', c['evaluation']['seed'])
        return json.loads((path / 'results.json').read_text())['metrics']['val']['f1']

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=config['seed']), direction='maximize')
    study.optimize(objective, n_trials=args.num_trials)
    write_json(output / 'best.json', {'score': study.best_value, 'params': study.best_trial.user_attrs['params']})


if __name__ == "__main__":
    main()
