import os
import delu
import json
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import evaluate, print_metrics, load_wrapper, average_metrics


def train_simple(
    data_path,
    exp_path,
    seed=0,
    eval_type='real',
    params=None,
    encoded_path=None
):
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
        print(f'loading real data: {encoded_path}')
        train_data = pd.read_csv(os.path.join(encoded_path, 'train.csv'))

        label_data = train_data.iloc[:, -1]
        train_data = train_data.iloc[:, :-1]
    else:
        raise "Choose eval method"

    val_data = pd.read_csv(os.path.join(encoded_path, 'val.csv'))
    test_data = pd.read_csv(os.path.join(encoded_path, 'test.csv'))

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
        }
    
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

    return {'metrics': avg_results, 'per_model': all_results}
