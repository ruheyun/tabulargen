import os
from copy import deepcopy
import delu
import json
import pandas as pd
from pprint import pprint
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from tabulargen.evaluation.metrics import evaluate, print_metrics, get_optimal_threshold_from_pr


def train_catboost(
    data_path,
    exp_path,
    seed=0,
    eval_type='real',
    params=None,
):
    delu.random.seed(seed)
    data_name = os.path.basename(data_path)

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    print('-' * 100)
    if eval_type == 'synthetic':
        print(f'loading synthetic data: {exp_path}')
        train_data = pd.read_csv(os.path.join(exp_path, 'reverse.csv'))

    elif eval_type == 'real':
        print(f'loading real data: {data_path}')
        train_data = pd.read_csv(os.path.join(data_path, f'{data_name}_train.csv'))
    else:
        raise ValueError("Choose real or synthetic evaluation")

    val_data = pd.read_csv(os.path.join(data_path, f'{data_name}_val.csv'))
    test_data = pd.read_csv(os.path.join(data_path, f'{data_name}_test.csv'))

    X = {
        'train': train_data.values[:, :-1],
        'val': val_data.values[:, :-1],
        'test': test_data.values[:, :-1],
    }

    le = LabelEncoder()
    y_train = le.fit_transform(train_data.values[:, -1])
    y_val = le.transform(val_data.values[:, -1])
    y_test = le.transform(test_data.values[:, -1])

    y = {
        'train': y_train,
        'val': y_val,
        'test': y_test,
    }

    print(f'Train size: {X["train"].shape}, Val size: {X["val"].shape}, Test size: {X["test"].shape}')

    if params is None:
        raise ValueError('CatBoost parameters must be provided explicitly')
    catboost_config = deepcopy(params)

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(info['n_num_features'], info['n_num_features'] + info['n_cat_features']))

    pprint(catboost_config, width=100)
    print('-' * 100)

    model = CatBoostClassifier(
        loss_function="MultiClass" if info['task_type'] == 'multiclass' else "Logloss",
        **catboost_config,
        eval_metric = 'AUC' if info['task_type'] != 'multiclass' else 'MacroF1',
        random_seed=seed,
        class_names=[str(i) for i in range(info['n_classes'])] if info['task_type'] == 'multiclass' else ["0", "1"],
        allow_writing_files=False
    )

    predict = (
        model.predict_proba
        if info['task_type'] == 'multiclass'
        else lambda x: model.predict_proba(x)[:, 1]
    )

    model.fit(X['train'], y['train'], eval_set=(X['val'], y['val']), verbose=100)

    predictions = {
        k: predict(v)
        for k, v in X.items()
    }

    # threshold = get_optimal_threshold_from_pr(y['val'], predictions['val'])

    results = {
        k: evaluate(y[k], predictions[k], info['task_type'])
        for k in predictions
    }

    print_metrics(results)

    return {"metrics": results, "per_model": {"catboost": results}}
