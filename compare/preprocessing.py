"""
Add a new split to the dataset to be used as a comparisson to Conditional Monge Gap"
"""

import scanpy as sc
import numpy as np

# Declare variables
train_split = 0.8
rng = np.random.default_rng(1337)

# Out of distribution conditions (i.e. conditions not present in the training set). Rest of conditions are used as train/test
ood = ["Dacinostat", "Givinostat", "Belinostat", "Hesperadin", "Quisinostat", "Alvespimycin", "Tanespimycin", "TAK-901", "Flavopiridol"]

# Load the dataset
adata = sc.read_h5ad("/home/thesis/ConditionalMongeGap/Datasets/sciplex_complete_middle_subset.h5ad")

# Add a new split to the dataset to be used as a comparisson to Conditional Monge Gap
adata.obs["compare_split"] = np.where(
    adata.obs["condition"].isin(ood),
    "ood",
    rng.choice(["train", "test"], p=[train_split, 1-train_split]),
) 

# Save the dataset
adata.write_h5ad("/home/thesis/ConditionalMongeGap/Datasets/sciplex_complete_middle_subset_compare.h5ad")