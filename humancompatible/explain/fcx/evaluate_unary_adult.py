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
    Load the Adult test split, the pre‑trained black-box classifier and FCX-VAE model,
    generate counterfactuals, and compute evaluation metrics.

    This wrapper runs the full evaluation pipeline for the Adult dataset, returning
    a dictionary of metrics including validity, causal feasibility, continuous and
    categorical proximity, and LOF anomaly scores.

    Args:
        base_data_dir (str): Path to the directory containing the preprocessed
            test split (.npy files) and JSON weight files.
        base_model_dir (str): Directory where the black-box model and VAE checkpoints
            are saved.
        dataset_name (str): Dataset identifier (default `'adult'`), used to name
            files like `{dataset_name}-test-set.npy` and `{dataset_name}.pth`.
        pth_name (str): Filename of the trained VAE checkpoint relative to
            `base_model_dir` (default `'models/adult_binary.pth'`).

    Returns:
        dict: A mapping `{ dataset_name: metrics_dict }`, where `metrics_dict` has keys
        `'validity'`, `'const-score'`, `'cont-prox'`, `'cat-prox'`, and `'LOF'`, each
        containing the computed score(s) for the FCX-VAE counterfactuals.
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
