from pathlib import Path
from seml.config import generate_configs, read_config
import sys
import torch

# Set the path to the folder containing the yaml file
PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PATH))
from chemCPA.experiments_run import ExperimentWrapper

# This is how seml loads the config file internally
seml_config, slurm_config, experiment_config = read_config(
   str(PATH) + "/manual_run.yaml"
)
configs = generate_configs(experiment_config)
config = configs[0]

# Create seml experiment
exp = ExperimentWrapper(init_all=False)
exp.seed = 1337
exp.init_dataset(**config["dataset"])

exp.init_drug_embedding(embedding=config["model"]["embedding"])
exp.init_model(
    hparams=config["model"]["hparams"],
    additional_params=config["model"]["additional_params"],
    load_pretrained=config["model"]["load_pretrained"],
    append_ae_layer=config["model"]["append_ae_layer"],
    enable_cpa_mode=config["model"]["enable_cpa_mode"],
    pretrained_model_path=config["model"]["pretrained_model_path"],
    pretrained_model_hashes=config["model"]["pretrained_model_hashes"],
)

exp.update_datasets()

# Train the model
print(f"Training the model on {'cuda' if torch.cuda.is_available() else 'cpu'}")
train_results = exp.train(**config["training"])
