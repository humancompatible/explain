# ------------------- LIBRARIES -------------------
from scripts.dataloader import DataLoader
from scripts.blackboxmodel import BlackBox
from scripts.evaluation_functions_census import *
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


def evaluate_census(
    base_data_dir: str = 'data/',
    base_model_dir: str = 'models/',
    dataset_name: str = 'census',
    pth_name: str = 'census-margin-0.764-feasibility-192.0-validity-29.0-epoch-25-fcx-unary.pth'
) -> dict:
    """
    Compute evaluation metrics for the Census unary FCX-VAE model.

    Returns:
        res (dict): Dictionary of computed metrics.
    """
    # ------------------- LOAD DATASET -------------------
    dataset = pd.read_csv(
        os.path.join(base_data_dir, dataset_name, 'census_data.csv')
    )
    params = {
        'dataframe': dataset.copy(),
        'continuous_features': [
            'age','wage_per_hour','capital_gains','capital_losses',
            'dividends_from_stocks','num_persons_worked_for_employer','weeks_worked_in_year'
        ],
        'outcome_name': 'income'
    }
    d = DataLoader(params)
    feat_to_change = d.get_indexes_of_features_to_vary([
        'age','class_of_worker','major_industry_code','major_occupation_code',
        'education','wage_per_hour','marital_status','full_parttime_employment_stat',
        'capital_gains','capital_losses','dividends_from_stocks','tax_filer_status',
        'd_household_family_stat','d_household_summary','num_persons_worked_for_employer',
        'family_members_under_18','citizenship','own_business_or_self_employed',
        'veterans_benefits','weeks_worked_in_year','year'
    ])

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
    encoded_size = 30
    sample_range = [1]
    div_case     = 1
    immutables   = True

    res = {dataset_name: {}}
    methods = {
        'FCX': os.path.join(base_model_dir, pth_name)
    }
    prefix_name = f"{dataset_name}-unary"

    # ------------------- COMPUTE EVALUATION METRICS -------------------
    res[dataset_name]['validity'] = compute_eval_metrics(
        immutables, methods, base_model_dir, encoded_size,
        pred_model, vae_test_dataset, d,
        normalise_weights, mad_feature_weights,
        div_case, 0, sample_range, 'census-validity'
    )
    res[dataset_name]['const-score'] = compute_eval_metrics(
        immutables, methods, base_model_dir, encoded_size,
        pred_model, vae_test_dataset, d,
        normalise_weights, mad_feature_weights,
        div_case, 1, sample_range, 'census-feasibility-score'
    )
    res[dataset_name]['cont-prox'] = compute_eval_metrics(
        immutables, methods, base_model_dir, encoded_size,
        pred_model, vae_test_dataset, d,
        normalise_weights, mad_feature_weights,
        div_case, 3, sample_range, 'census-cont-proximity-score'
    )
    res[dataset_name]['cat-prox'] = compute_eval_metrics(
        immutables, methods, base_model_dir, encoded_size,
        pred_model, vae_test_dataset, d,
        normalise_weights, mad_feature_weights,
        div_case, 4, sample_range, 'census-cat-proximity-score'
    )
    res[dataset_name]['LOF'] = compute_eval_metrics(
        immutables, methods, base_model_dir, encoded_size,
        pred_model, vae_test_dataset, d,
        normalise_weights, mad_feature_weights,
        div_case, 8, sample_range,
        'census-unary-lof-score', prefix_name=prefix_name
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_dir',  type=str, default='data/')
    parser.add_argument('--base_model_dir', type=str, default='models/')
    parser.add_argument('--dataset_name',   type=str, default='census')
    parser.add_argument('--pth_name',       type=str, default='census-margin-0.764-feasibility-192.0-validity-29.0-epoch-25-fcx-unary.pth')
    args = parser.parse_args()

    metrics = evaluate_census(
        base_data_dir=args.base_data_dir,
        base_model_dir=args.base_model_dir,
        dataset_name=args.dataset_name,
        pth_name=args.pth_name
    )
    print(metrics)