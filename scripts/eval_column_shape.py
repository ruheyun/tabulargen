import pandas as pd
from sdmetrics.reports.single_table import QualityReport
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import dump_json, load_json


def get_metadata(data, data_name):
    
    metadata = {'columns': {}}

    for i, column in enumerate(data.columns):
        metadata['columns'][column] = {
            'sdtype': 'numerical' if data[column].dtype != 'object' else 'categorical'
        }

    dump_json(metadata, os.path.join('data', data_name, 'metadata.json'))

    return metadata


def metrics(real_data, synthetic_data, metadata):
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata=metadata, verbose=False)

    print(report.get_properties())
    print('===============================================')

    # print(report.get_score())
    # print('===============================================')

    # print(report.get_details(property_name='Column Shapes'))
    # print('===============================================')

    # print(report.get_details(property_name='Column Pair Trends'))
    # print('===============================================')

    # fig = report.get_visualization(property_name='Column Pair Trends')
    # fig.save('exp/cpt.png')

if __name__ == '__main__':

    data_name = 'adult'

    real_data = pd.read_csv(f'data/{data_name}/{data_name}_train.csv')
    synthetic_data = pd.read_csv(f'exp/{data_name}/reverse.csv')
    metadata = load_json(f'data/{data_name}/metadata.json')

    metrics(real_data, synthetic_data, metadata)
