import os
import delu
import json
import pandas as pd
from pprint import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from tabulargen.data.encoding import load_wrapper
from tabulargen.evaluation.metrics import evaluate, print_metrics, average_metrics


def train_simple(
    data_path,
    exp_path,
    seed=0,
    eval_type='real',
    params=None,
    encoded_path=None,
):
    encoded_path = Path(encoded_path or exp_path)
    delu.random.seed(seed)

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    print('-' * 100)
    if eval_type == 'synthetic':
        print(f'loading synthetic data: {exp_path}')
        train_data = pd.read_csv(os.path.join(exp_path, 'reverse.csv'))

        data_wrapper = load_wrapper(encoded_path / 'data_wrapper.pkl')

        label_wrapper = load_wrapper(encoded_path / 'label_wrapper.pkl')

        label_data = label_wrapper.transform(train_data.iloc[:, -1].values)
        train_data = data_wrapper.transform(train_data.iloc[:, :-1])

    elif eval_type == 'real':
        print(f'loading real data: {exp_path}')
        train_data = pd.read_csv(encoded_path / 'train.csv')

        label_data = train_data.iloc[:, -1]
        train_data = train_data.iloc[:, :-1]
    else:
        raise ValueError("Choose real or synthetic evaluation")

    val_data = pd.read_csv(encoded_path / 'val.csv')
    test_data = pd.read_csv(encoded_path / 'test.csv')

    X = {
        'train': train_data.values,
        'val': val_data.values[:, :-1],
        'test': test_data.values[:, :-1],
    }

    y = {
        'train': label_data.values.reshape(-1),
        'val': val_data.values[:, -1],
        'test': test_data.values[:, -1],
    }

    print(f'Train size: {X["train"].shape}, Val size: {X["val"].shape}, Test size: {X["test"].shape}')

    print('-' * 100)

    models = {
            "tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=seed),
            "rf": RandomForestClassifier(max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1),
            "lr": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed),
            "mlp": MLPClassifier(max_iter=1000, random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=16),
            # "knn": KNeighborsClassifier(n_neighbors=5, weights='distance')
        }

    if params:
        unknown = set(params) - set(models)
        if unknown:
            raise ValueError(f"Unknown simple models: {unknown}")
        for name, options in params.items():
            models[name].set_params(**options)
            models[name].set_params(random_state=seed)
    all_results = {}
    for model_name, model in models.items():
        print(model.__class__.__name__)

        predict = (
            model.predict_proba
            if info['task_type'] == 'multiclass'
            else lambda x: model.predict_proba(x)[:, 1]
        )

        model.fit(X['train'], y['train'])

        predictions = {
            k: predict(v)
            for k, v in X.items()
        }

        results = {
            k: evaluate(y[k], predictions[k], info['task_type'])
            for k in predictions
        }

        all_results[model_name] = results

        print_metrics(results)
        print()

    avg_results = average_metrics(all_results)
    print('Average results')
    print_metrics(avg_results)

    return {"metrics": avg_results, "per_model": all_results}
