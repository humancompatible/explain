# ------------------- LIBRARIES -------------------

from scripts.dataloader import DataLoader
from scripts.fcx_vae_model import FCX_VAE
from scripts.blackboxmodel import BlackBox
from scripts.evaluation_functions_adult import *
from scripts.helpers import *

import sys
import random
import pandas as pd
import numpy as np
import json
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

torch.manual_seed(10000000)


def evaluate_adult(
    base_data_dir: str = 'data/',
    base_model_dir: str = 'models/',
    dataset_name: str = 'adult',
    pth_name:str = 'models/adult_binary.pth'
) -> dict:
    """
    Compute evaluation metrics for the Adult dataset.

    This function will:
      1. Load the Adult test split from `base_data_dir`.
      2. Load the pre-trained black‑box classifier from `base_model_dir/{dataset_name}.pth`.
      3. Load the trained FCX-VAE model from `base_model_dir/{pth_name}`.
      4. Generate counterfactuals on the test set.
      5. Compute and return the following metrics:
         - **validity**: fraction of counterfactuals that successfully flip the classifier.
         - **const-score**: causal feasibility score.
         - **cont-prox**: continuous feature L1 proximity.
         - **cat-prox**: categorical feature L1 proximity.
         - **LOF**: Local Outlier Factor anomaly score.

    Args:
        base_data_dir (str): Directory containing preprocessed `.npy` data arrays.
        base_model_dir (str): Directory containing the trained `.pth` models.
        dataset_name (str): Name of the dataset (default: 'adult').
        pth_name (str): Filename of the trained VAE model (relative to `base_model_dir`).

    Returns:
        dict: {
                dataset_name: {
                    'validity': ...,
                    'const-score': ...,
                    'cont-prox': ...,
                    'cat-prox': ...,
                    'LOF': ...,
            }
    """
    # ------------------- LOAD DATASET -------------------
    dataset = load_adult_income_dataset()
    params = {'dataframe': dataset.copy(), 'continuous_features': ['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)

    vae_test_dataset = np.load(base_data_dir + dataset_name + '-test-set.npy')
    vae_test_dataset = vae_test_dataset[vae_test_dataset[:,-1] == 0, :]
    vae_test_dataset = vae_test_dataset[:, :-1]

    with open(base_data_dir + dataset_name + '-normalise_weights.json') as f:
        normalise_weights = {int(k):v for k,v in json.load(f).items()}

    with open(base_data_dir + dataset_name + '-mad.json') as f:
        mad_feature_weights = json.load(f)

    # ------------------- LOAD BLACKBOX MODEL -------------------
    data_size = len(d.encoded_feature_names)
    pred_model = BlackBox(data_size)
    path = base_model_dir + dataset_name + '.pth'
    pred_model.load_state_dict(torch.load(path))
    pred_model.eval()

    # ------------------- PARAMETERS -------------------
    encoded_size = 10
    sample_range = [1]
    div_case = 1
    feat_to_change = d.get_indexes_of_features_to_vary([
        'age','hours_per_week','workclass','education','marital_status','occupation'
    ])
    immutables = True

    res = {dataset_name: {}}
    methods = {
        'FCX': base_model_dir + pth_name
    }

    prefix_name = 'adult-binary'

    # ------------------- COMPUTE EVALUATION METRICS -------------------
    res[dataset_name]['validity'] = compute_eval_metrics_adult(
        immutables,
        methods,
        base_model_dir,
        encoded_size,
        pred_model,
        vae_test_dataset,
        d,
        normalise_weights,
        mad_feature_weights,
        div_case,
        0,
        sample_range,
        'adult-validity'
    )
    res[dataset_name]['const-score'] = compute_eval_metrics_adult(
        immutables,
        methods,
        base_model_dir,
        encoded_size,
        pred_model,
        vae_test_dataset,
        d,
        normalise_weights,
        mad_feature_weights,
        div_case,
        2,
        sample_range,
        'adult-binary-feasibility-score'
    )
    res[dataset_name]['cont-prox'] = compute_eval_metrics_adult(
        immutables,
        methods,
        base_model_dir,
        encoded_size,
        pred_model,
        vae_test_dataset,
        d,
        normalise_weights,
        mad_feature_weights,
        div_case,
        3,
        sample_range,
        'adult-cont-proximity-score'
    )
    res[dataset_name]['cat-prox'] = compute_eval_metrics_adult(
        immutables,
        methods,
        base_model_dir,
        encoded_size,
        pred_model,
        vae_test_dataset,
        d,
        normalise_weights,
        mad_feature_weights,
        div_case,
        4,
        sample_range,
        'adult-cat-proximity-score'
    )
    res[dataset_name]['LOF'] = compute_eval_metrics_adult(
        immutables,
        methods,
        base_model_dir,
        encoded_size,
        pred_model,
        vae_test_dataset,
        d,
        normalise_weights,
        mad_feature_weights,
        div_case,
        7,
        sample_range,
        'adult-binary-lof-score',
        prefix_name=prefix_name
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_dir', type=str, default='data/')
    parser.add_argument('--base_model_dir', type=str, default='models/')
    parser.add_argument('--dataset_name', type=str, default='adult')
    parser.add_argument('--pth_name', type=str, default='pth_name')

    args = parser.parse_args()

    # call wrapper function
    metrics = evaluate_adult(
        base_data_dir=args.base_data_dir,
        base_model_dir=args.base_model_dir,
        dataset_name=args.dataset_name,
        pth_name=args.pth_name
    )
    print(metrics)
