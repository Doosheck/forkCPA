import scanpy as sc
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="create_dataset.py",
    description="Create a dataset from a raw data file.",
)

parser.add_argument(
    "--path-data",
    dest="path",
    type=str,
    help="Path to the projects folder.",
)


parser.add_argument(
    "--name-save",
    dest="name",
    type=str,
    help="Path to the projects folder.",
)

args = parser.parse_args()

# Read data
adata = sc.read_h5ad(args.path + "ConditionalMongeGap/Datasets/sciplex_complete_middle_subset.h5ad")

# Create dataset. Select a single cell line and dose
adata = adata[
    (adata.obs["cell_type"] == "A549")
    & (adata.ob["dose"].isin([10000, 0]))
]

# Save dataset
adata.write_h5ad(args.path + "ConditionalMongeGap/Datasets/" + f"{name}.h5ad")