# How to use ChemCPA
## Environment setup
The environment setup provided by the repository takes a very long time 
1. Create a new conda environment
```conda create --name=chemcpa  python=3.7```
1. Activate the environment
```conda activate chemcpa```
1. Install required packages via pip (not best practice, but works)
```pip install -r requirements.txt```

## Experiment setup
We are going to use ```manual_seml_sweep.py``` to run experiments. The setup for the experiment can be found in ```manual_run.taml```, and the program expects the configuration file in the same path as itself.

In the content of the manual run we must specify the following:
1. ```dataset.data_params.dataset_path```: path to stored dataset.
1. ```dataset.data_params.split_key```: key in the andata object that specifies the split (train/test/out-of-distr.).
1. ```training.save_chekpoints``` and ```training.save_dir```: if and where to store the checkpoints

Additionally, we can change the specific configuration of the model, optimizer, and training loop. 
