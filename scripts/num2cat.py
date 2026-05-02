import pandas as pd
import string

df = pd.read_csv('data/bank/bank.csv')

threshold = 15  # 唯一值阈值

prefix_list = list(string.ascii_lowercase)  # ['a','b','c',...]

prefix_idx = 0

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        prefix = prefix_list[prefix_idx]
        prefix_idx += 1
        unique_vals = df[col].dropna().unique()
        
        if len(unique_vals) < threshold and all(float(x).is_integer() for x in unique_vals):
            # 转成字符串类别 p0, p1...
            mapping = {val: f'{prefix}{i}' for i, val in enumerate(sorted(unique_vals))}
            df[col] = df[col].map(mapping)

df.to_csv('data/bank.csv', index=False)