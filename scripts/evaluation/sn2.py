import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

   
original_data = pd.read_csv(os.path.join('data/adult/adult_train.csv'))
syn_data = pd.read_csv(os.path.join('data/adult/adult_train.csv'))
syn_DP_data = pd.read_csv(os.path.join('data/adult/adult_train.csv'))
syn_IDP_data = pd.read_csv(os.path.join('data/adult/adult_train.csv'))

col = "workclass"
original_data["Dataset"] = "Original Data"
syn_data["Dataset"] = "Non-DP Synthetic Data"
syn_DP_data["Dataset"] = "DP Synthetic Data"
syn_IDP_data["Dataset"] = "Importance DP Synthetic Data"

combined_df = pd.concat(
    [
        original_data[[col, "Dataset"]],
        syn_data[[col, "Dataset"]],
        syn_DP_data[[col, "Dataset"]],
        syn_IDP_data[[col, "Dataset"]],
    ]
)

sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(
    data=combined_df,
    x=col,
    hue="Dataset",
    ax=ax,
    palette=["C0", "C1", "C2", "C3"]
)
ax.set_title(f"Data Distribution for Column: {col}")
plt.legend()
plt.show()