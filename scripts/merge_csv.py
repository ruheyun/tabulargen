import pandas as pd

df1 = pd.read_csv('data/default/default_train.csv')
df2 = pd.read_csv('data/default/default_test.csv')

df = pd.concat([df1, df2], axis=0)

df.to_csv('data/default/default.csv', index=False)