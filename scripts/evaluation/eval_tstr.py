import os
import delu
import json
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from catboost import CatBoostClassifier, CatBoostRegressor
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import evaluate, print_metrics, load_wrapper, average_metrics
from scripts.data.data_encode import DataWrapper, LabelWrapper 


def train_tstr(
    data_path,
    exp_path,
    seed=0,
    eval_type='real',
    params_path=None,
    model_names=None,
):
    delu.random.seed(seed)

    if model_names is None:
        model_names = ['catboost', 'tree', 'rf', 'lr', 'mlp']

    catboost_params = {}
    if 'catboost' in model_names:
        if params_path is None:
            raise ValueError('CatBoost parameters path must be provided')
        catboost_params = json.loads(Path(params_path).read_text())

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    print('-' * 100)
    if eval_type == 'synthetic':
        print(f'loading synthetic data: {exp_path}')
        train_data = pd.read_csv(os.path.join(exp_path, 'reverse.csv'))

    elif eval_type == 'real':
        print(f'loading real data: {data_path}')
        train_data = pd.read_csv(os.path.join(data_path, f'{info['name']}_train.csv'))

    else:
        raise ValueError(f'Unknown eval_type: {eval_type}')

    val_data = pd.read_csv(os.path.join(data_path, f'{info['name']}_val.csv'))
    test_data = pd.read_csv(os.path.join(data_path, f'{info['name']}_test.csv'))
    
    label_wrapper = LabelWrapper(task=info['task_type'])
    label_wrapper.fit(train_data.iloc[:, -1].to_numpy())

    frames = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    if any(name != 'catboost' for name in model_names):
        eval_wrapper = DataWrapper(
            num_encoder='standard', cat_encoder='oht', seed=42
        )
        eval_wrapper.fit(
            train_data.iloc[:, :-1],
            num_features=info['n_num_features'],
        )
        X_simple = {
            split: eval_wrapper.transform(frame.iloc[:, :-1]).to_numpy(dtype='float64')
            for split, frame in frames.items()
        }

    X_catboost = {
        split: frame.iloc[:, :-1].copy()
        for split, frame in frames.items()
    }

    y = {
        split: label_wrapper.transform(frame.iloc[:, -1].to_numpy()).to_numpy().reshape(-1)
        for split, frame in frames.items()
    }


    print(f'Train size: {frames["train"].shape}, Val size: {frames["val"].shape}, Test size: {frames["test"].shape}')

    print('-' * 100)

    if info['task_type'] == 'regression':
        models = {
            "catboost": CatBoostRegressor(
                **catboost_params, 
                eval_metric='RMSE', 
                random_seed=seed, 
                allow_writing_files=False
            ),
            "tree": DecisionTreeRegressor(max_depth=8, random_state=seed),
            "rf": RandomForestRegressor(max_depth=15, random_state=seed),
            "lr": Ridge(max_iter=500, random_state=seed),
            "mlp": MLPRegressor(max_iter=500, random_state=seed)
        }
    else:
        models = {
            "catboost": CatBoostClassifier(
                loss_function="MultiClass" if info['task_type'] == 'multiclass' else "Logloss",
                **catboost_params,
                eval_metric = 'AUC' if info['task_type'] != 'multiclass' else 'TotalF1',
                random_seed=seed,
                class_names=[str(i) for i in range(info['n_classes'])] if info['task_type'] == 'multiclass' else ["0", "1"],
                allow_writing_files=False,
            ),
            "tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=seed),
            "rf": RandomForestClassifier(max_depth=15, min_samples_leaf=5, random_state=seed, n_jobs=-1),
            "lr": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed),
            "mlp": MLPClassifier(max_iter=1000, random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=16),
        }

    if model_names is not None:
        unknown = set(model_names) - set(models)
        if unknown:
            raise ValueError(f'Unknown models: {sorted(unknown)}')
        models = {name: models[name] for name in model_names}

    all_results = {}
    for model_name, model in models.items():
        print(model.__class__.__name__)
        X = X_catboost if model_name == 'catboost' else X_simple

        if model_name == 'catboost':
            model.fit(X['train'], y['train'], eval_set=(X['val'], y['val']), verbose=0)
        else:
            model.fit(X['train'], y['train'])

        predict = (
            model.predict
            if info['task_type'] == 'regression'
            else model.predict_proba
            if info['task_type'] == 'multiclass'
            else lambda x: model.predict_proba(x)[:, 1]
        )

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
