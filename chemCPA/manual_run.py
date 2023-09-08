from pathlib import Path
from pprint import pprint
import sys
import numpy as np

# from seml.config import generate_configs, read_config
from gori import Configurator
from config import generate_configs, read_config

print(sys.path)

# from chemCPA.experiments_run import ExperimentWrapper
from experiments_run import ExperimentWrapper


if __name__ == "__main__":
    exp = ExperimentWrapper(init_all=False)

    # this is how seml loads the config file internally
    assert Path(
        "/mnt/storage/kasperek/sc-transformer/finetuning_num_genes.json"
    ).exists(), "config file not found"
    seml_config, slurm_config, experiment_config = read_config(
        "/mnt/storage/kasperek/sc-transformer/finetuning_num_genes.json"
    )
    seml_config, slurm_config, experiment_config = read_config(
        "/mnt/storage/kasperek/sc-transformer/finetuning_num_genes.json"
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
    # setup the torch DataLoader
    exp.update_datasets()

    exp.train(**args["training"])
