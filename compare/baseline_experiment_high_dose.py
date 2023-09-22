from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import seml
import os

from matplotlib import pyplot as plt
matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-talk")
matplotlib.rcParams["font.family"] = "monospace"
plt.rcParams["savefig.facecolor"] = "white"
sns.set_context("poster")
pd.set_option("display.max_columns", 15)

"""
Chem_CPA experiments are susceptible to randomness. In particular, we need to run the same experiment multiple times to ensure best possible evaluation.
All the experiments are run with the same hyperparameters, but different random seeds.
The results are then averaged over the runs (bagging)
"""
for num in range(0, 5):
    os.system(
                "/dss/dsshome1/0A/di93hoq/miniconda3/envs/run_chem_cpa/bin/python /mnt/forkCPA/compare/train.py"
                + "--path=/mnt/forkCPA/"
                + "--conig=/mnt/forkCPA/compare/baseline_experiment_highest_dose.yaml"
                + f"--save=/mnt/forkCPA/compare/checkpoints/trained_model_{num}.pt"
            )


