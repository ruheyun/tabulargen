import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


real = pd.read_csv(f"data/adult/adult_train.csv").select_dtypes("number").corr()
syn  = pd.read_csv(f"exp/adult/syn-dp.csv").select_dtypes("number").corr()
cols = real.columns.intersection(syn.columns)
D = (real[cols] - syn[cols]).abs().values

plt.figure(figsize=(3, 3.5))
plt.imshow(D, cmap=LinearSegmentedColormap.from_list("wr", ["white", "#B2182B"]),
           vmin=0, vmax=np.nanmax(D), interpolation="nearest")
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.title(f"Importance-DP-10")  # Importance-DP
# plt.savefig("corr_diff.png", dpi=300, bbox_inches="tight")
plt.show()