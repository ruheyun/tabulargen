import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(data_path):
    
    data_name = os.path.basename(data_path)
    df = pd.read_csv(os.path.join(data_path, f'{data_name}.csv'))
    target_col = df.columns[-1].strip()
    X, y = df.drop(columns=[target_col]), df[target_col]

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    X_reordered = X[num_cols + cat_cols]
    df_final = pd.concat([X_reordered, y], axis=1)

    idx_train, idx_temp = train_test_split(
        df.index, test_size=0.40, random_state=42, stratify=y
    )

    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.60, random_state=42, stratify=y[idx_temp]
    )

    df_final.loc[idx_train].reset_index(drop=True).to_csv(os.path.join(data_path, f'{data_name}_train.csv'), index=False)
    df_final.loc[idx_val].reset_index(drop=True).to_csv(os.path.join(data_path, f'{data_name}_val.csv'), index=False)
    df_final.loc[idx_test].reset_index(drop=True).to_csv(os.path.join(data_path, f'{data_name}_test.csv'), index=False)

    info = {
        'name': f'{data_name}',
        'task_type': ('binclass' if len(y.unique()) == 2 else 'multiclass') if len(y.unique()) < 100 else 'regression',
        'n_classes': len(y.unique()),
        'n_num_features': len(num_cols),
        'n_cat_features': len(cat_cols),
        'train_size': len(idx_train),
        'val_size': len(idx_val),
        'test_size': len(idx_test)
    }

    with open(os.path.join(data_path, 'info.json'), 'w') as f:
        json.dump(info, f)

    print(f'Spliting done!')


if __name__ == '__main__':
    data_path = 'data/adult'
    data_split(data_path)
