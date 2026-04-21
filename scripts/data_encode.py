import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, LabelEncoder


class DataWrapper:
    def __init__(self, num_encoder="quantile", cat_encoder='alb', seed=42):
        self.num_encoder = num_encoder
        self.cat_encoder = cat_encoder
        self.seed = seed
        self.raw_dim = None
        self.raw_columns = None
        self.all_distinct_values = {}
        self.num_normalizer = {}
        self.num_dim = 0
        self.columns = []
        self.col_dim = []
        self.col_dtype = {}

    def fit(self, dataframe, num_features):
        self.raw_dim = dataframe.shape[1]
        self.raw_columns = dataframe.columns

        for col in self.raw_columns[:num_features]:
            col_data = dataframe.loc[pd.notna(dataframe[col])][col]
            self.col_dtype[col] = col_data.dtype
            if self.num_encoder == "quantile":
                self.num_normalizer[col] = QuantileTransformer(
                    output_distribution='normal',  # normal: [-inf, inf]  uniform: [0, 1]
                    n_quantiles=max(min(len(col_data) // 30, 1000), 10),
                    subsample=1000000000,
                    random_state=self.seed, )
            elif self.num_encoder == "standard":
                self.num_normalizer[col] = StandardScaler()
            elif self.num_encoder == "minmax":
                self.num_normalizer[col] = MinMaxScaler()
            else:
                raise ValueError(f"Unknown num encoder: {self.num_encoder}")
            self.num_normalizer[col].fit(col_data.values.reshape(-1, 1))
            self.columns.append(col)
            self.num_dim += 1
            self.col_dim.append(1)

        for col in self.raw_columns[num_features:]:
            col_data = dataframe.loc[pd.notna(dataframe[col])][col]
            self.col_dtype[col] = col_data.dtype
            distinct_values = col_data.unique()
            distinct_values.sort()
            self.all_distinct_values[col] = distinct_values
            self.columns.append(col)
            if self.cat_encoder == 'alb':
                dim = max(1, int(np.ceil(np.log2(len(distinct_values)))))
            elif self.cat_encoder == 'oht':
                dim = len(distinct_values)
            else:
                raise ValueError(f"Unknown cat encoder: {self.cat_encoder}")
            self.col_dim.append(dim)

    def transform(self, data):
        reorder_data = data[self.columns].values
        norm_data = []
        for i, col in enumerate(self.columns):
            col_data = reorder_data[:, i]
            if col in self.all_distinct_values.keys():
                col_data = self.CatValsToNum(col, col_data)
                if self.cat_encoder == 'alb':
                    col_data = self.ValsToBit(col_data.reshape(-1, 1), self.col_dim[i])
                elif self.cat_encoder == 'oht':
                    onehot = np.zeros((len(col_data), self.col_dim[i]))
                    onehot[np.arange(len(col_data)), col_data] = 1
                    col_data = onehot  
                norm_data.append(col_data)
            elif col in self.num_normalizer.keys():
                norm_data.append(self.num_normalizer[col].transform(col_data.reshape(-1, 1)).reshape(-1, 1))
        norm_data = np.concatenate(norm_data, axis=1)
        norm_data = norm_data.astype(np.float32)
        return norm_data

    def ReOrderColumns(self, data: pd.DataFrame):
        ndf = pd.DataFrame([])
        for col in self.raw_columns:
            ndf[col] = data[col]
        return ndf

    def GetColData(self, data, col_id):
        col_index = np.cumsum(self.col_dim)
        col_data = data.copy()
        if col_id == 0:
            return col_data[:, :col_index[0]]
        else:
            return col_data[:, col_index[col_id - 1]:col_index[col_id]]

    def ValsToBit(self, values, bits):
        bit_values = np.zeros((values.shape[0], bits))
        for i in range(values.shape[0]):
            bit_val = np.mod(np.right_shift(int(values[i]), list(reversed(np.arange(bits)))), 2)
            bit_values[i, :] = bit_val
        return bit_values

    def BitsToVals(self, bit_values):
        bits = bit_values.shape[1]
        values = bit_values.astype(int)
        values = values * (2 ** np.array(list((reversed(np.arange(bits))))))
        values = np.sum(values, axis=1)
        return values

    def CatValsToNum(self, col, values):
        num_values = pd.Categorical(values, categories=self.all_distinct_values[col]).codes
        return num_values

    def NumValsToCat(self, col, values):
        cat_values = np.zeros_like(values).astype(object)
        values = np.clip(values, 0, len(self.all_distinct_values[col]) - 1)
        for i, val in enumerate(values):
            cat_values[i] = self.all_distinct_values[col][int(val)]
        return cat_values

    def ReverseToOrdi(self, data):
        reverse_data = []

        for i, col in enumerate(self.columns):
            col_data = self.GetColData(data, i)
            if col in self.all_distinct_values.keys():
                if self.cat_encoder == 'alb':
                    col_data = np.round(col_data)
                    col_data = self.BitsToVals(col_data)
                elif self.cat_encoder == 'oht':
                    col_data = np.argmax(col_data, axis=1)

                col_data = col_data.astype(np.int32)
            else:
                col_data = self.num_normalizer[col].inverse_transform(col_data.reshape(-1, 1))
                if self.col_dtype[col] == np.int32 or self.col_dtype[col] == np.int64:
                    col_data = np.round(col_data).astype(int)

            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def ReverseToCat(self, data):
        reverse_data = []
        for i, col in enumerate(self.columns):
            col_data = data[:, i]
            if col in self.all_distinct_values.keys():
                col_data = self.NumValsToCat(col, col_data)
            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def Reverse(self, data):
        data = self.ReverseToOrdi(data)
        data = self.ReverseToCat(data)
        data = pd.DataFrame(data, columns=self.columns)
        return self.ReOrderColumns(data)

    def RejectSample(self, sample):
        all_index = set(range(sample.shape[0]))
        allow_index = set(range(sample.shape[0]))
        for i, col in enumerate(self.columns):
            if col in self.all_distinct_values.keys():
                allow_index = allow_index & set(np.where(sample[:, i] < len(self.all_distinct_values[col]))[0])
                allow_index = allow_index & set(np.where(sample[:, i] >= 0)[0])
        reject_index = all_index - allow_index
        allow_index = np.array(list(allow_index))
        reject_index = np.array(list(reject_index))
        return allow_index, reject_index


class LabelWrapper:
    def __init__(self, task='binclass'):
        self.task = task
        self.encoder = None

    def fit(self, y):
        y = np.array(y)

        if self.task == 'binclass' or self.task == 'multiclass':
            self.encoder = LabelEncoder()
            self.encoder.fit(y)

        elif self.task == 'regression':
            self.encoder = StandardScaler()
            self.encoder.fit(y.reshape(-1, 1))

        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def transform(self, y):
        y = np.array(y)

        if self.task == 'binclass' or self.task == 'multiclass':
            return self.encoder.transform(y)

        elif self.task == 'regression':
            return self.encoder.transform(y.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, y):
        y = np.array(y)

        if self.task == 'binclass' or self.task == 'multiclass':
            return self.encoder.inverse_transform(y.astype(int))

        elif self.task == 'regression':
            return self.encoder.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        