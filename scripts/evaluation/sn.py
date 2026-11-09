import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

        
original_data = pd.read_csv(os.path.join('data/adult/adult_train.csv'))


sns.set_theme()
col = "age"
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=original_data, x=col, label="Original Data", ax=ax, color='C0', fill=True)
# sns.kdeplot(data=synt_dataset, x=col, label="Non-DP FinDiff Synthetic Data", ax=ax, color='C2', fill=True)
# sns.kdeplot(data=synt_dataset_DP, x=col, label="DP FinDiff Synthetic Data", ax=ax, color='C1', fill=True)
ax.set_title(f"Distribution Comparison for Column: {col}")
plt.legend()
plt.show()