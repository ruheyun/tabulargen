import pandas as pd

def evaluate_minority_ratio(real_data, synthetic_data, target_column):
    
    class_counts = real_data[target_column].value_counts()
    minority_class = class_counts.idxmin()
    
    print(f"真实数据类别分布:\n{class_counts}")
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

    data_name = 'adult'

    real_data = pd.read_csv(f'data/{data_name}/{data_name}_train.csv')
    synthetic_data = pd.read_csv(f'exp/{data_name}/ctgan/reverse-1.csv')

    target_column = real_data.columns[-1]
    evaluate_minority_ratio(
        real_data=real_data, 
        synthetic_data=synthetic_data, 
        target_column=target_column
    )