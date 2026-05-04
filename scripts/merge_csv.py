import pandas as pd

# df1 = pd.read_csv('data/cardio/train.csv')
# df2 = pd.read_csv('data/cardio/val.csv')

# df = pd.concat([df1, df2], axis=0)

# df.to_csv('data/cardio/cardio.csv', index=False)


df = pd.read_csv('data/market/market.csv', sep=';')
df.to_csv('data/market/market.csv', index=False)