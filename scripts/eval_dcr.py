import os
import pandas as pd
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import load_json


def dcr_base(real_data, synthetic_data, metadata):
    score = DCRBaselineProtection.compute_breakdown(
        real_data=real_data,          
        synthetic_data=synthetic_data,  
        metadata=metadata,
        num_rows_subsample=1000,
        num_iterations=5               
    )

    print(f'dcr_baseline score: {score}')

    return score


def dcr_over(real_train_data, real_val_data, synthetic_data, metadata):
    score = DCROverfittingProtection.compute_breakdown(
        real_training_data=real_train_data,    
        synthetic_data=synthetic_data,        
        real_validation_data=real_val_data,    
        metadata=metadata,
        num_rows_subsample=1000,
        num_iterations=5
    )

    print(f'dcr_overfit score: {score}')

    return score


if __name__ == '__main__':

    data_name = 'adult'

    real_train_data = pd.read_csv(os.path.join('data', data_name, f'{data_name}_train.csv'))
    real_val_data = pd.read_csv(os.path.join('data', data_name, f'{data_name}_val.csv'))
    synthetic_data = pd.read_csv(os.path.join('exp', data_name, 'reverse.csv'))

    metadata = load_json(f'data/{data_name}/metadata.json')

    dcr_base(real_train_data, synthetic_data, metadata)
    dcr_over(real_train_data, real_val_data, synthetic_data, metadata)
