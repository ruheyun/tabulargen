import pandas as pd

name = 'house'
df1 = pd.read_csv(f'data/{name}/train.csv')
df2 = pd.read_csv(f'data/{name}/val.csv')
df3 = pd.read_csv(f'data/{name}/test.csv')

df = pd.concat([df1, df2, df3], axis=0)

df.to_csv(f'data/{name}/{name}.csv', index=False)


# df = pd.read_csv('data/market/market.csv', sep=';')
# df.to_csv('data/market/market.csv', index=False)