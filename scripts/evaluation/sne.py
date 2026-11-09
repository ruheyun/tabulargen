import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


# merge original categorical and numerical attributes
real_dataset_evaluation = pd.read_csv(os.path.join('exp/adult/run_00/encoded/train.csv'))
synt_dataset_evaluation = pd.read_csv(os.path.join('exp/adult/unnondp.csv'))
synt_dataset_DP_evaluation = pd.read_csv(os.path.join('exp/adult/unsyn-idp.csv'))

# determine TSNE embeddings
real_dataset_embedded = TSNE(n_components=2, random_state=42).fit_transform(real_dataset_evaluation)
synthetic_dataset_embedded = TSNE(n_components=2, random_state=42).fit_transform(synt_dataset_evaluation)
synthetic_dataset_DP_embedded = TSNE(n_components=2, random_state=42).fit_transform(synt_dataset_DP_evaluation)


fig, ax = plt.subplots(1, 3, figsize=(16, 6))  # Adjust the figure size as needed
# set main title for the entire figure
fig.suptitle("t-SNE Embeddings of Original and Synthetic Datasets", fontsize=24)
# Scatter plot with enhanced aesthetics and plotted above the grid
ax[0].scatter(real_dataset_embedded[:, 0], real_dataset_embedded[:, 1],
              color='C0', alpha=0.6, edgecolors='black', s=10, linewidth=0.3, zorder=5)
ax[1].scatter(synthetic_dataset_embedded[:, 0], synthetic_dataset_embedded[:, 1],
              color='C2', alpha=0.6, edgecolors='black', s=10, linewidth=0.3, zorder=5)
ax[2].scatter(synthetic_dataset_DP_embedded[:, 0], synthetic_dataset_DP_embedded[:, 1],
              color='C1', alpha=0.6, edgecolors='black', s=10, linewidth=0.3, zorder=5)

# remove ticks and labels for a cleaner look
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel('')
    a.set_ylabel('')
# add titles for each subplot
ax[0].set_title("Original Data", fontsize=16)
ax[1].set_title("Non-DPSynthetic Data", fontsize=16)
ax[2].set_title("DP Synthetic Data", fontsize=16)
plt.tight_layout()
plt.show()