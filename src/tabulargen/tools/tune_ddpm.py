"""Tune each trial in its own run, using the same explicit stage interfaces."""
import argparse
from copy import deepcopy
from pathlib import Path
import optuna

from tabulargen.cli import run, write_json
from tabulargen.config import resolve_config
from tabulargen.layout import RunPaths
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New directory for trial artifacts')
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--sample-seeds', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args()
    config = resolve_config(args.config)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new tuning output directory')
    output.mkdir(parents=True)

    def objective(trial):
        trial_config = deepcopy(config)
        trial_config['experiment']['path'] = str(output / f'trial_{trial.number}')
        trial_config['train']['lr'] = trial.suggest_float('lr', 1e-5, 0.003, log=True)
        run(trial_config, ['encode', 'train'])
        scores = []
        for seed in args.sample_seeds:
            trial_config['sample']['seed'] = seed
            run(trial_config, ['sample', 'eval'])
            e = trial_config['evaluation']
            path = RunPaths(Path(trial_config['experiment']['path'])).evaluation(e['mode'], seed, e['model'], e['seed'])
            scores.append(json.loads((path / 'results.json').read_text())['metrics']['val']['roc_auc'])
        trial.set_user_attr('config', trial_config)
        return sum(scores) / len(scores)

    if config['evaluation']['mode'] != 'synthetic':
        raise ValueError('Diffusion tuning requires synthetic evaluation')
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=config['seed']), direction='maximize')
    study.optimize(objective, n_trials=args.num_trials)
    write_json(output / 'best.json', {'score': study.best_value, 'trial': study.best_trial.number,
                                     'config': study.best_trial.user_attrs['config']})


if __name__ == "__main__":
    main()
