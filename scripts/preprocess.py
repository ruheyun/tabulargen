import os
import json
import pickle
import pandas as pd
import numpy as np
from data_encode import DataWrapper, LabelWrapper


def data_process(data_path, exp_path, num_encoder='quantile', cat_encoder='alb'):
    out_path = os.path.join(exp_path, 'preprocess')
    os.makedirs(out_path, exist_ok=True)
    data_name = os.path.basename(data_path)

    df_train = pd.read_csv(os.path.join(data_path, f'{data_name}_train.csv'))
    df_val = pd.read_csv(os.path.join(data_path, f'{data_name}_val.csv'))
    df_test = pd.read_csv(os.path.join(data_path, f'{data_name}_test.csv'))

    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)     

    train_wrapper = DataWrapper(num_encoder=num_encoder, cat_encoder=cat_encoder)
    train_wrapper.fit(df_train.iloc[:, :-1], num_features=info['n_num_features'])

    X_train_encoding = train_wrapper.transform(df_train.iloc[:, :-1])
    X_val_encoding = train_wrapper.transform(df_val.iloc[:, :-1])
    X_test_encoding = train_wrapper.transform(df_test.iloc[:, :-1])

    label_wrapper = LabelWrapper(task=info['task_type'])
    label_wrapper.fit(df_train.iloc[:, [-1]])

    y_train_encoding = label_wrapper.transform(df_train.iloc[:, [-1]])
    y_val_encoding = label_wrapper.transform(df_val.iloc[:, [-1]])
    y_test_encoding = label_wrapper.transform(df_test.iloc[:, [-1]])

    df_train_encoding = pd.DataFrame(np.concatenate([X_train_encoding, y_train_encoding], axis=1))
    df_val_encoding = pd.DataFrame(np.concatenate([X_val_encoding, y_val_encoding], axis=1))
    df_test_encoding = pd.DataFrame(np.concatenate([X_test_encoding, y_test_encoding], axis=1))

    # num_cols = [f'num_{i}' for i in range(info['n_num_features'])]
    # cat_cols = [f'cat_{i}' for i in range(info['n_cat_features'])]
    # y_cols = ['label']
    # cols = num_cols + cat_cols + y_cols

    # df_train_encoding.columns =  cols       
    # df_val_encoding.columns = cols        
    # df_test_encoding.columns = cols   

    info['n_features'] = len(X_train_encoding.loc[0])

    df_train_encoding.to_csv(os.path.join(out_path, f'{data_name}_train.csv'), index=False)
    df_val_encoding.to_csv(os.path.join(out_path, f'{data_name}_val.csv'), index=False)
    df_test_encoding.to_csv(os.path.join(out_path, f'{data_name}_test.csv'), index=False)

    with open(os.path.join(out_path, 'info.json'), 'w') as f:
        json.dump(info, f)

    with open(os.path.join(out_path, "data_wrapper.pkl"), 'wb') as f:
        pickle.dump(train_wrapper, f)

    with open(os.path.join(out_path, "label_wrapper.pkl"), 'wb') as f:
        pickle.dump(label_wrapper, f)


if __name__ == '__main__':
    data_path = 'data/adult'
    exp_path = 'exp/adult'
    data_process(data_path, exp_path)
