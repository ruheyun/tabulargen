import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class TabularDataset(Dataset):

    def __init__(self, data_path, type='train'):

        df = pd.read_csv(os.path.join(data_path, f'{type}.csv'))

        with open(os.path.join(data_path, 'info.json'), 'r') as f:
            info = json.load(f)

        if info['task_type'] == 'binclass':
            label_dtype = torch.float32
        else:
            label_dtype = torch.long

        self.y = torch.tensor(df['label'].values, dtype=label_dtype)
        self.X = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)

        self.X_dim = self.X.shape[1]

        assert self.X_dim == info['encoded_dim'], ('data dim false!')

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


