import jax.numpy as jnp
import json

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
    "--path",
    dest="path",
    type=str,
    help="Path to the projects folder.",
)

parser.add_argument(
    "--num_trainings",
    dest="num_trainings",
    type=int,
    help="Number of trainings to perform.",
)

parser.add_argument(
    "--config",
    dest="config",
    type=str,
    help="Path to the config file.",
)

parser.add_argument(
    "--save",
    dest="save",
    type=str,
    help="Path and name of file to save the model.",
)
fargs = parser.parse_args()

exp = ExperimentWrapper(init_all=False)
# this is how seml loads the config file internally
assert Path(
    fargs.config
).exists(), "config file not found"
seml_config, slurm_config, experiment_config = read_config(
   fargs.config
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

for num in range(fargs.num_trainings):
    save_resuls = f"{fargs.save[:-3]}_{num}_dict.json"
    train_results = exp.train(**args["training"], save_name=f"{fargs.save[:-3]}_{num}.pt")
    print(f"Training {num} finished. Saving results as {save_resuls}")
    json.dump(train_results, open(save_resuls, "w"))