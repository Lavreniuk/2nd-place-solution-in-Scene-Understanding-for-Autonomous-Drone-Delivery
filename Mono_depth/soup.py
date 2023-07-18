import argparse
import itertools
import os
import subprocess
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import linalg
from torch.nn import functional as F


def mix_many_models():
    model_paths = [
        os.path.join(args.model_path, file_name)
        for file_name in os.listdir(args.model_path)
    ]

    alphas = np.arange(0, 1.1, 0.1)
    alpha_combinations = list(itertools.product(alphas, repeat=len(model_paths)))
    alpha_combinations = [
        alpha_combination
        for alpha_combination in alpha_combinations
        if sum(alpha_combination) == 1.0
    ]
    #alpha_combinations = [[0.25, 0.25, 0.25, 0.25]]
    alpha_combinations = [[1/len(model_paths)]*len(model_paths)]
    for i, alpha_combination in enumerate(alpha_combinations):
        print(f"Combination {i + 1} of {len(alpha_combinations)}")
        print(f"Alphas: {alpha_combination}")

        # Load models and calculate weighted average
        theta = {}
        for j, model_path in enumerate(model_paths):
            alpha = alpha_combination[j]
            theta_j = torch.load(model_path)["model"]

            for key in theta_j.keys():
                if key not in theta:
                    theta[key] = alpha * theta_j[key]
                else:
                    theta[key] += alpha * theta_j[key]

        # update the model acccording to the new weights
        
        #torch.load(model_path)
        torch.save(
            {
                "model": theta,
            },
            "/home/nick/shortcuts/monodepth3_checkpoints/soup/res.pt",
        )
        #subprocess.call(["python", "local_evaluation.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()

    mix_many_models()
    
