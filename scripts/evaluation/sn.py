import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

        
original_data = pd.read_csv(os.path.join('data/bank/bank_train.csv'))
syn_data = pd.read_csv(os.path.join('exp/bank/nondp.csv'))
# syn_DP_data = pd.read_csv(os.path.join('exp/adult/syn-dp.csv'))
syn_IDP_data = pd.read_csv(os.path.join('exp/bank/syn-dp.csv'))


sns.set_theme()
col = "estimated_salary"
fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=original_data, x=col, label="Original Data", ax=ax, color='C0', fill=True)
sns.kdeplot(data=syn_data, x=col, label="Non-DP Synthetic Data", ax=ax, color='C1', fill=True)
# sns.kdeplot(data=syn_DP_data, x=col, label="DP Synthetic Data", ax=ax, color='C2', fill=True)
sns.kdeplot(data=syn_IDP_data, x=col, label="Importance DP Synthetic Data", ax=ax, color='C2', fill=True)
ax.set_title(f"Distribution Comparison for Column: {col}")
# ax.set_xlim(-5, 5)   # 按实际分位数调整
plt.legend()
plt.savefig("sn1.png", dpi=300, bbox_inches="tight")
plt.show()
