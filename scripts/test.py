import pandas as pd
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

real_data = pd.read_csv(r'data\king\king_train.csv')

synt_data = pd.read_csv(r'exp\king\tvae\reverse.csv')

synt_data = synt_data[list(real_data.columns)]

synt_data.to_csv(os.path.join(r'exp\king\tvae', 'reverse.csv'), index=False, header=True)