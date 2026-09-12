import os
import json
import pickle
import pandas as pd
import delu
from tabulargen.data.encoding import DataWrapper, LabelWrapper
from tabulargen.training.privacy import dp_histogram
from tabulargen.artifacts import ENCODED_FILES


def data_process(data_path, exp_path, num_encoder='quantile', cat_encoder='alb',
                 seed=0, histogram_epsilon=0.1, histogram_delta=1e-5):
    if any(os.path.exists(os.path.join(exp_path, name)) for name in ENCODED_FILES):
        raise FileExistsError(
            f'Encoding artifacts or model already exist in {exp_path}. '
            'Choose a new experiment.path to avoid replacing a model\'s encoding.'
        )
    delu.random.seed(seed)
    os.makedirs(exp_path, exist_ok=True)
    data_name = os.path.basename(data_path)

    df_train = pd.read_csv(os.path.join(data_path, f'{data_name}_train.csv'))
    df_val = pd.read_csv(os.path.join(data_path, f'{data_name}_val.csv'))
    df_test = pd.read_csv(os.path.join(data_path, f'{data_name}_test.csv'))

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    if info['task_type'] not in ('binclass', 'multiclass'):
        raise ValueError('The generation pipeline currently supports classification only')

    train_wrapper = DataWrapper(num_encoder=num_encoder, cat_encoder=cat_encoder, seed=seed)
    train_wrapper.fit(df_train.iloc[:, :-1], num_features=info['n_num_features'])

    X_train_encoding = train_wrapper.transform(df_train.iloc[:, :-1])
    X_val_encoding = train_wrapper.transform(df_val.iloc[:, :-1])
    X_test_encoding = train_wrapper.transform(df_test.iloc[:, :-1])

    label_wrapper = LabelWrapper(task=info['task_type'])
    label_wrapper.fit(df_train.iloc[:, -1].values)

    y_train_encoding = label_wrapper.transform(df_train.iloc[:, -1].values)
    y_val_encoding = label_wrapper.transform(df_val.iloc[:, -1].values)
    y_test_encoding = label_wrapper.transform(df_test.iloc[:, -1].values)

    df_train_encoding = pd.concat([X_train_encoding, y_train_encoding], axis=1)
    df_val_encoding = pd.concat([X_val_encoding, y_val_encoding], axis=1)
    df_test_encoding = pd.concat([X_test_encoding, y_test_encoding], axis=1)

    info['raw_feature_count'] = df_train.shape[1] - 1
    info['encoded_dim'] = X_train_encoding.shape[1]
    info['encoding'] = {'num_encoder': num_encoder, 'cat_encoder': cat_encoder, 'seed': seed,
                        'histogram_epsilon': histogram_epsilon, 'histogram_delta': histogram_delta}
    info['y_name'] = [df_train.columns[-1]]

    num_cols = [f'num_{i}' for i in range(info['n_num_features'])]
    cat_cols = [f'cat_{i}' for i in range(info['encoded_dim'] - info['n_num_features'])]
    y_cols = ['label']
    cols = num_cols + cat_cols + y_cols

    df_train_encoding.columns =  cols
    df_val_encoding.columns = cols
    df_test_encoding.columns = cols

    df_train_encoding.to_csv(os.path.join(exp_path, 'train.csv'), index=False)
    df_val_encoding.to_csv(os.path.join(exp_path, 'val.csv'), index=False)
    df_test_encoding.to_csv(os.path.join(exp_path, 'test.csv'), index=False)

    dp_p_y = dp_histogram(
        y_train_encoding,
        num_classes=info['n_classes'],
        epsilon=histogram_epsilon,
        delta=histogram_delta
    )

    info['dp_p_y'] = dp_p_y.tolist()

    label_counts = y_train_encoding.value_counts().sort_index()
    origin_p_y = (label_counts / label_counts.sum()).values

    info['origin_p_y'] = origin_p_y.tolist()

    with open(os.path.join(exp_path, 'info.json'), 'w') as f:
        json.dump(info, f)

    with open(os.path.join(exp_path, "data_wrapper.pkl"), 'wb') as f:
        pickle.dump(train_wrapper, f)

    with open(os.path.join(exp_path, "label_wrapper.pkl"), 'wb') as f:
        pickle.dump(label_wrapper, f)
