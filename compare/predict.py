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

parser = ArgumentParser(
    prog="predict.py",
    description="Predict the perturbation effect of a drug. And compute metrics.",
)

parser.add_argument(
    "--path",
    dest="path",
    type=str,
    help="Path to the forkCPA project folder.",
    default="/home/thesis/forkCPA/",
)

parser.add_argument(
    "--utils",
    dest="utils",
    type=str,
    help="Path to the utils folder.",
    default="/home/thesis/ConditionalMongeGap/",
)

parser.add_argument(
    "--model",
    dest="model",
    type=str,
    help="Path to the model.",
    default="/home/thesis/forkCPA/compare/checkpoints/"
)

parser.add_argument(
    "--save",
    dest="save",
    type=str,
    help="Path and name of file to save the results.",
    default="/home/thesis/forkCPA/compare/results.csv"
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
    model_paths = [fargs.model]

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
    with open(f"{fargs.save}", 'w') as f:
        w = csv.DictWriter(f, ["name", "type", "r2", "mae", "sinkhorn_source_target", "sinkhorn_target_pred", "mmd_source_target", "mmd_target_pred", "fid_source_target", "fid_target_pred", "e_source_target", "e_target_pred"])
        w.writeheader()
        print(
            f"\n{'Condition':25s}{'':5s}" +
            f"{'Type':10s}{'':5s}{'Disentanglement'}{'':5s}{'r2':>15s}{'':5s}{'mae':>15s}" +
            f"{'':5s}{'SINK(S,T)':>15s}{'':5s}{'SINK(T,P)':>15s}" +
            f"{'':5s}{'MMD(S,T)':>15s}{'':5s}{'MMD(T,P)':>15s}"+
            f"{'':5s}{'FID(S,T)':>15s}{'':5s}{'FID(S,P)':>12s}" +
            f"{'':5s}{'E(S,T)':>15s}{'':5s}{'E(T,P)':>15s}"
        )
        print("-"*240)
        for type_ in ["test", "ood"]:
            prediction, embeddings = model.predict(
                genes=exp.datasets[type_].genes,
                drugs_idx=exp.datasets[type_].drugs_idx,
                dosages=exp.datasets[type_].dosages,
                covariates=exp.datasets[type_].covariates
            )

            # If the return has mean and std concatenated
            prediction_mean = prediction.detach().numpy()[:, 0:2000]
            prediction_std = prediction.detach().numpy()[:, 2000:4000]  

            # If the return has alternating mean and std (this is NOT how the model was saved)
            # prediction_mean = prediction.detach().numpy()[:, 0::2]
            # prediction_std = prediction.detach().numpy()[:, 1::2]

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
                
                print(
                    ("{:25s}{:5s}{:10s}{:5s}" + "{:>15.3f}{:5s}" * 10 +"{:>15.3f}").format(
                        name,
                        '',
                        type_,
                        '',
                        train_results["perturbation disentanglement"][-1],
                        '',
                        results['r2'],
                        '',
                        results["mae"],
                        '',
                        results['sinkhorn_source_target'],
                        '',
                        results['sinkhorn_target_pred'],
                        '' ,
                        results['mmd_source_target'],
                        '',
                        results['mmd_target_pred'],
                        '',
                        results['fid_target_pred'],
                        '',
                        results['fid_source_target'],
                        '',
                        results['e_source_target'],
                        '',
                        results['e_target_pred']
                    )
                )
                
                w.writerow(results)
            
            print("-"*240)

print(f"Skipped {len(skipped)} drugs: {skipped}")