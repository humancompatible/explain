# ------------------- LIBRARIES -------------------
from scripts.dataloader import DataLoader
from scripts.blackboxmodel import BlackBox
from scripts.evaluation_functions_law import *
from scripts.helpers import *

import os
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


def evaluate_law(
    base_data_dir: str = 'data/',
    base_model_dir: str = 'models/',
    dataset_name: str = 'law',
    pth_name: str = 'law-margin-0.344-feasibility-87.0-validity-76.0-epoch-25-fcx-binary.pth'
) -> dict:
    """
    Compute evaluation metrics for the Law dataset.

    Returns:
        res (dict): Dictionary of computed metrics.
    """
    # ------------------- LOAD DATASET -------------------
    dataset = pd.read_csv(
        os.path.join(base_data_dir, dataset_name, 'bar_pass_prediction_v2.csv')
    )
    params = {
        'dataframe': dataset.copy(),
        'continuous_features': ['lsat','decile1b','decile3','ugpa','zfygpa','zgpa'],
        'outcome_name': 'pass_bar'
    }
    d = DataLoader(params)
    feat_to_change = d.get_indexes_of_features_to_vary(
        ['lsat','decile1b','decile3','ugpa','zfygpa','zgpa','fulltime','fam_inc','tier']
    )

    vae_test_dataset = np.load(
        os.path.join(base_data_dir, dataset_name, f"{dataset_name}-test-set.npy")
    )
    vae_test_dataset = vae_test_dataset[vae_test_dataset[:,-1] == 0, :]
    vae_test_dataset = vae_test_dataset[:, :-1]

    with open(
        os.path.join(base_data_dir, dataset_name, f"{dataset_name}-normalise_weights.json")
    ) as f:
        normalise_weights = {int(k): v for k, v in json.load(f).items()}

    with open(
        os.path.join(base_data_dir, dataset_name, f"{dataset_name}-mad.json")
    ) as f:
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
    div_case     = 1
    immutables   = True

    res = {dataset_name: {}}
    methods = {
        'FCX': os.path.join(base_model_dir, pth_name)
    }
    prefix_name = f"{dataset_name}-binary"

    # ------------------- COMPUTE EVALUATION METRICS -------------------
    res[dataset_name]['validity'] = compute_eval_metrics(
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
        'law-validity'
    )
    res[dataset_name]['const-score'] = compute_eval_metrics(
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
        'law-feasibility-score'
    )
    res[dataset_name]['cont-prox'] = compute_eval_metrics(
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
        'law-cont-proximity-score'
    )
    res[dataset_name]['cat-prox'] = compute_eval_metrics(
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
        'law-cat-proximity-score'
    )
    res[dataset_name]['LOF'] = compute_eval_metrics(
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
        'law-lof',
        prefix_name=prefix_name
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_dir',  type=str, default='data/')
    parser.add_argument('--base_model_dir', type=str, default='models/')
    parser.add_argument('--dataset_name',   type=str, default='law')
    parser.add_argument('--pth_name',       type=str, default='law-margin-0.344-feasibility-87.0-validity-76.0-epoch-25-fcx-binary.pth')
    args = parser.parse_args()

    metrics = evaluate_law(
        base_data_dir=args.base_data_dir,
        base_model_dir=args.base_model_dir,
        dataset_name=args.dataset_name,
        pth_name=args.pth_name
    )
    print(metrics)
