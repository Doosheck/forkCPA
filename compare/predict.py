import csv
import torch
import sys
import numpy as np
import jax.numpy as jnp
import os
import json

from pathlib import Path
from pprint import pprint
from seml.config import generate_configs, read_config
from chemCPA.experiments_run import ExperimentWrapper
from chemCPA.model import ComPert
from argparse import ArgumentParser
sys.path.append("/dss/dsshome1/0A/di93hoq/forkCPA/")
from notebooks.utils import compute_pred

parser = ArgumentParser(
    prog="predict.py",
    description="Predict the perturbation effect of a drug. And compute metrics.",
)

parser.add_argument(
    "--path",
    dest="path",
    type=str,
    help="Path to the forkCPA project folder.",
)

parser.add_argument(
    "--utils",
    dest="utils",
    type=str,
    help="Path to the utils folder.",
)

parser.add_argument(
    "--model",
    dest="model",
    type=str,
    help="Path to the model.",
)

parser.add_argument(
    "--save",
    dest="save",
    type=str,
    help="Path and name of file to save the results.",
)

fargs = parser.parse_args()

exp = ExperimentWrapper(init_all=False)
# this is how seml loads the config file internally
assert Path(
    fargs.path + "manual_run.yaml"
).exists(), "config file not found"
seml_config, slurm_config, experiment_config = read_config(
   fargs.path + "manual_run.yaml"
)
# we take the first config generated
configs = generate_configs(experiment_config)
if len(configs) > 1:
    print("Careful, more than one config generated from the yaml file")
args = configs[0]
pprint(args)

exp.seed = 1337
# loads the dataset splits
exp.init_dataset(**args["dataset"])

exp.init_drug_embedding(embedding=args["model"]["embedding"])
exp.init_model(
    hparams=args["model"]["hparams"],
    additional_params=args["model"]["additional_params"],
    load_pretrained=args["model"]["load_pretrained"],
    append_ae_layer=args["model"]["append_ae_layer"],
    enable_cpa_mode=args["model"]["enable_cpa_mode"],
    pretrained_model_path=args["model"]["pretrained_model_path"],
    pretrained_model_hashes=args["model"]["pretrained_model_hashes"],
)

exp.update_datasets()

# Load the utilities from the main project
sys.path.append(fargs.utils)
print(fargs.utils)
from utils import calculate_metrics

if os.path.isdir(fargs.model):
    models = [model for model in os.listdir(fargs.model) if model.endswith(".pt")]
else:
    models = [fargs.model]

with open(f"{fargs.save}", 'w') as f:
    w = csv.DictWriter(f, ["model", "name", "type", "disentanglement", "r2", "mae", "sinkhorn_source_target", "sinkhorn_target_pred", "mmd_source_target", "mmd_target_pred", "fid_source_target", "fid_target_pred", "e_source_target", "e_target_pred"])
    w.writeheader()
    
    for model_name in models:
        with open(f"{fargs.model}{model_name[:-3]}_dict.json", 'rb') as f:
            train_results = json.load(f)

        model = torch.load(fargs.model + model_name)

        (
            state_dict,
            cov_adv_state_dicts,
            cov_emb_state_dicts,
            init_args,
            history,

        ) = model

        model = ComPert(
                **init_args, drug_embeddings=exp.drug_embeddings
        )

        model = model.eval()
        skipped = []
    

        for type_ in ["test", "ood"]:
            prediction, embeddings = model.predict(
                genes=exp.datasets[type_].genes,
                drugs_idx=exp.datasets[type_].drugs_idx,
                dosages=exp.datasets[type_].dosages,
                covariates=exp.datasets[type_].covariates
            )

            drug_r2, _ = compute_pred(
                model,
                exp.datasets[type_]
            )

            print(f"Drug R2 {type_}: {drug_r2}")
            
            prediction_mean = prediction.detach().numpy()[:, 0:2000]
            prediction_std = prediction.detach().numpy()[:, 2000:4000]  

            for name in np.unique(exp.datasets[type_].drugs_names):
                section = (exp.datasets[type_].drugs_names == name)
                source = jnp.asarray(exp.datasets["test_control"].genes[0:len(section)])
                target = jnp.asarray(exp.datasets[type_].genes[section])
                predicted=jnp.asarray(prediction_mean[section])

                if (target.shape[0] == 1) or (predicted.shape[0] == 1):
                    skipped.append(name)
                    continue

                results = calculate_metrics(
                    name=name,
                    type=type_,
                    source=source,
                    target=target,
                    predicted=predicted,
                    epsilon=0.1,
                    epsilon_mmd=100
                )

                results["model"] = model_name
                results["disentanglement"] = train_results[0]["perturbation disentanglement"]
                
                w.writerow(results)

print(f"Skipped {len(skipped)} drugs: {skipped}")