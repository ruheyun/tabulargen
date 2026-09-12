import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import load_json


def csv2npy(data_name, split):

    df = pd.read_csv(os.path.join('data', data_name, f'{data_name}_{split}.csv'))

    metadata = load_json(f'data/{data_name}/metadata.json')

    columns_info = metadata["columns"]
    all_columns = list(columns_info.keys())
    label_col = all_columns[-1]

    num_cols = []
    cat_cols = []

    for col, info in columns_info.items():

        if col == label_col:
            continue

        if info["sdtype"] == "numerical":
            num_cols.append(col)

        elif info["sdtype"] == "categorical":
            cat_cols.append(col)

    X_num = df[num_cols]
    X_cat = df[cat_cols]
    y = df[label_col]

    if len(num_cols):
        X_num_np = X_num.to_numpy(dtype=np.float32)
        np.save(os.path.join('exp', 'npy', data_name, f'X_num_{split}.npy'), X_num_np)

    if len(cat_cols):
        X_cat_np = X_cat.astype(str).to_numpy()
        np.save(os.path.join('exp', 'npy', data_name, f'X_cat_{split}.npy'), X_cat_np)

    y_np = y.to_numpy()
    np.save(os.path.join('exp', 'npy', data_name, f'y_{split}.npy'), y_np)


if __name__ == '__main__':
    data_name = 'adult'
    split = ['train', 'val', 'test']

    os.makedirs(os.path.join('exp', 'npy', data_name), exist_ok=True)

    for sp in split:
        csv2npy(data_name, sp)
