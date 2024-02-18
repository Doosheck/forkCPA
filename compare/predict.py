import scanpy as sc
import sys
import torch

from chemCPA.model import ComPert
from chemCPA.train import compute_prediction
from notebooks.utils import repeat_n
import argparse
import pandas as pd

evaluator_path = "/home/icb/gori.camps/ConditionalOT_Perturbations/src/"
sys.path.append(evaluator_path)
from evaluator import calculate_metrics

# Set model path and dataset path
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="The path of the model to load",
)
argparser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="The path of the dataset to load",
)

fargs = argparser.parse_args()

#Load the model
model = torch.load(
    fargs.model_path,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

(
    state_dict,
    cov_adv_state_dicts,
    cov_emb_state_dicts,
    init_args,
    history,

) = model

model = ComPert(
        **init_args,
)

# Get the data to evaluate on
adata = sc.read_h5ad(
    fargs.dataset_path
)

"""
Compute the predictions.
We hardcode the cell line and condition for now.
Since we want to average over the conditions, we will simply use their idx, without caring
what the actual trearment is.
"""
cell_line_dict = {
    "A549": torch.tensor([1, 0, 0]),
    "K562": torch.Tensor([0, 1, 0]),
    "MCF7": torch.Tensor([0, 0, 1]),
}
MAX_DOSE = 10000
for cell_line_name, cell_line_code in cell_line_dict.items():
    metrics = {}
    for condition_idx in range(len(adata.obs['condition'].unique())):
        
        condition_name = adata.obs['condition'].unique()[condition_idx]
        split = adata[
                (adata.obs['cell_type'] == cell_line_name) &
                (adata.obs['condition'] == condition_name)
        ].obs["split_ood_finetuning"]
        
        gex_target = torch.Tensor(
            adata[
                (adata.obs['cell_type'] == cell_line_name) &
                (adata.obs['condition'] == condition_name)
            ].X.A
        )

        gex_source = torch.Tensor(
            adata[
                (adata.obs['cell_type'] == cell_line_name) &
                (adata.obs['condition'] == 'control')
            ].X.A
        )[:gex_target.shape[0]]
        
        cell_line_repeated = [
            repeat_n(
                cell_line_code,
                gex_source.shape[0]
            )
        ]

        condition_repeated = (
            repeat_n(torch.Tensor([condition_idx]), gex_source.shape[0]).squeeze().to(torch.long),
            repeat_n(torch.Tensor([MAX_DOSE]), gex_source.shape[0]).squeeze().to(torch.long)
        )

        # ChemCPA predicts a gaussian for each gene, we take the mean as the prediction
        prediction, _= compute_prediction(
            model,
            genes=gex_source,
            emb_covs=cell_line_repeated,
            emb_drugs=condition_repeated,
        )
        
        metrics[f"{cell_line_name}_{condition_name}"] = calculate_metrics(
            gex_source,
            gex_target,
            prediction
        )
        
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df["split"] = split
        
        metrics_df.to_csv("metrics_chemcpa.csv", mode='a', header=False)
