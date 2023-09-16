import jax.numpy as jnp
import os

from pathlib import Path
from pprint import pprint
from seml.config import generate_configs, read_config
from chemCPA.experiments_run import ExperimentWrapper
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="train.py",
    description="Train ChemCPA",
)

parser.add_argument(
    "--path-data",
    dest="path",
    type=str,
    help="Path to the projects folder.",
)

fargs = parser.parse_args()

exp = ExperimentWrapper(init_all=False)
# this is how seml loads the config file internally
assert Path(
    fargs.path + "forkCPA/manual_run.yaml"
).exists(), "config file not found"
seml_config, slurm_config, experiment_config = read_config(
   fargs.path + "forkCPA/manual_run.yaml"
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

train_results = exp.train(**args["training"])