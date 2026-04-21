import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pathlib import Path


def data_split(data_path):
    """
    第一步：切出 70% 训练集（保留 30% 待分）
    第二步：剩下的 30% 对半分（50% 给 eval，50% 给 test）
    第三步：保存为 CSV
    """

    df = pd.read_csv(data_path)
    target_col = df.columns[-1].strip()
    X, y = df.drop(columns=[target_col]), df[target_col]

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    X_reordered = X[num_cols + cat_cols]
    df_final = pd.concat([X_reordered, y], axis=1)

    idx_train, idx_temp = train_test_split(
        df.index, test_size=0.30, random_state=42, stratify=y
    )

    idx_eval, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp]
    )

    pt = Path(data_path)
    df_final.loc[idx_train].reset_index(drop=True).to_csv(str(pt.with_name(f"{pt.stem}_train.csv")), index=False)
    df_final.loc[idx_eval].reset_index(drop=True).to_csv(str(pt.with_name(f"{pt.stem}_eval.csv")), index=False)
    df_final.loc[idx_test].reset_index(drop=True).to_csv(str(pt.with_name(f"{pt.stem}_test.csv")), index=False)

    info = {
        'name': f'{pt.stem}',
        'task_type': ('binclass' if len(y.unique()) == 2 else 'multiclass') if len(y.unique()) < 50 else 'regression',
        'n_num_features': len(num_cols),
        'n_cat_features': len(cat_cols),
        'train_size': len(idx_train),
        'val_size': len(idx_eval),
        'test_size': len(idx_test)
    }

    with open(str(pt.with_name('info.json')), 'w') as f:
        json.dump(info, f)

    print(f'Split done!')


if __name__ == '__main__':
    data_path = 'data/adult/adult.csv'
    data_split(data_path)
