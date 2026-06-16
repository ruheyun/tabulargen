import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from utils import dump_json, load_json


def get_metadata(data, data_name):
    """
    对数据集的每个特征类型进行判断，分为两类：数值型、类别型，保存为json文件
    """
    
    metadata = {'columns': {}}

    for i, column in enumerate(data.columns):
        metadata['columns'][column] = {
            'sdtype': 'numerical' if data[column].dtype != 'object' else 'categorical'
        }

    dump_json(metadata, os.path.join('data', data_name, 'metadata.json'))

    return metadata


def column_shape(real_data, synthetic_data, metadata):
    """
    Column shapes score, Column pair trends for real and synthetic datasets
    """

    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata=metadata, verbose=False)

    print(report.get_properties())
    # print('===============================================')

    # print(report.get_score())
    # print('===============================================')

    # print(report.get_details(property_name='Column Shapes'))
    # print('===============================================')

    # print(report.get_details(property_name='Column Pair Trends'))
    # print('===============================================')

    # fig = report.get_visualization(property_name='Column Pair Trends')
    # fig.save('exp/cpt.png')


def dcr_base(real_data, synthetic_data, metadata):
    """
    DCR baseline protection metric
    """

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
    """
    DCR overfitting protection metric
    """

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


def evaluate_minority_ratio(real_data, synthetic_data, target_column):
    """
    Percentage of the minority class in synthetic data
    """
    
    class_counts = real_data[target_column].value_counts()
    minority_class = class_counts.idxmin()
    
    # print(f"真实数据类别分布:\n{class_counts}")
    # print(f"少数类是: '{minority_class}'\n")
    
    orig_total = len(real_data)
    orig_minority_count = (real_data[target_column] == minority_class).sum()
    orig_ratio = orig_minority_count / orig_total * 100
    
    synth_total = len(synthetic_data)
    synth_minority_count = (synthetic_data[target_column] == minority_class).sum()
    synth_ratio = synth_minority_count / synth_total * 100
    
    diff = abs(synth_ratio - orig_ratio)
    
    print(f"评估结果:")
    print(f"   Original 少数类占比 : {orig_ratio:.2f}%")
    print(f"   Synthetic 少数类占比: {synth_ratio:.2f}%")
    print(f"   绝对误差 (Abs Error): {diff:.2f}%")
    
    # if diff < 1.0:
    #     print("分布还原度极好")
    # elif diff < 5.0:
    #     print("分布还原度可接受")
    # else:
    #     print("分布偏差较大，可能存在 Mode Collapse")
        
    return {
        'minority_class': minority_class,
        'original_ratio': orig_ratio,
        'synthetic_ratio': synth_ratio,
        'absolute_error': diff
    }


if __name__ == '__main__':
    # 数据集
    data_name = 'adult'

    real_train_data = pd.read_csv(os.path.join('data', data_name, f'{data_name}_train.csv'))
    real_val_data = pd.read_csv(os.path.join('data', data_name, f'{data_name}_val.csv'))
    real_test_data = pd.read_csv(os.path.join('data', data_name, f'{data_name}_val.csv'))
    val_test_data = pd.concat([real_val_data, real_test_data], ignore_index=True)
    synthetic_data = pd.read_csv(os.path.join('exp', data_name, 'reverse.csv'))
    synthetic_data.columns = real_train_data.columns

    # get_metadata(real_data, data_name)
    metadata = load_json(f'data/{data_name}/metadata.json')

    column_shape(real_train_data, synthetic_data, metadata)
    print('==================================================')

    dcr_base(real_train_data, synthetic_data, metadata)
    print('==================================================')

    dcr_over(real_train_data, val_test_data, synthetic_data, metadata)
    print('==================================================')

    evaluate_minority_ratio(real_train_data, synthetic_data, real_train_data.columns[-1])
