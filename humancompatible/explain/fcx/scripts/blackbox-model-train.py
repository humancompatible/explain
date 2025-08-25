#!/usr/bin/env python
# blackbox‑model‑train.py

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from dataloader import DataLoader
from blackboxmodel import BlackBox
from helpers import load_adult_income_dataset

def train_blackbox(
    dataset_name: str,
    base_data_dir: str = '../../data/',
    base_model_dir: str = '../models/',
    seed: int = 10000000,
    epochs: int = 100,
    batch_size: int = None,
    learning_rate: float = None
) -> BlackBox:
    """
    Train (or fine-tune) a BlackBox classifier on a specified dataset and save its weights.

    This function will:
      1. Load the chosen dataset’s training and validation splits.
      2. Instantiate a BlackBox MLP with input dimension matching the encoded features.
      3. Configure an optimizer (Adam or SGD) and cross‑entropy loss.
      4. Optionally balance the census training set by down‑sampling.
      5. Run a standard training loop for `epochs` epochs, reporting training accuracy.
      6. Evaluate on the held-out validation split, reporting validation accuracy.
      7. Save the trained model’s `state_dict()` to `{base_model_dir}/{dataset_name}.pth`.

    Args:
        dataset_name (str):
            Which dataset to train on. One of:
            - 'adult'
            - 'census'
            - 'law'
            - 'folktables_adult'
        base_data_dir (str):
            Path to the root data directory containing `{dataset_name}-*.npy` or CSV files.
        base_model_dir (str):
            Directory in which to save the final model checkpoint.
        seed (int):
            Random seed for both NumPy and PyTorch for reproducibility.
        epochs (int):
            Number of training epochs to run.
        batch_size (int, optional):
            Mini-batch size for training (defaults to dataset‑specific default).
        learning_rate (float, optional):
            Learning rate for the optimizer (defaults to dataset‑specific default).

    Returns:
        BlackBox:
            The trained BlackBox model instance. Its weights are also saved to disk.

    Raises:
        ValueError:
            If `dataset_name` is not one of the supported options.
    """
    # reproducibility
    torch.manual_seed(seed)

    # set defaults if not provided
    bs = batch_size
    lr = learning_rate

    # — Dataset-specific setup —
    if dataset_name == 'adult':
        dataset = load_adult_income_dataset()
        params = {
            'dataframe': dataset.copy(),
            'continuous_features': ['age', 'hours_per_week'],
            'outcome_name': 'income'
        }
        d = DataLoader(params)
        inp_shape = len(d.encoded_feature_names)
        pred_model = BlackBox(inp_shape)
        bs = bs or 32
        lr = lr or 0.01
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        train_path = os.path.join(base_data_dir, f"{dataset_name}-train-set.npy")
        val_path   = os.path.join(base_data_dir, f"{dataset_name}-val-set.npy")

    elif dataset_name == 'census':
        dataset = pd.read_csv(os.path.join(base_data_dir, 'census', 'census_data.csv'))
        params = {
            'dataframe': dataset.copy(),
            'continuous_features': [
                'age', 'wage_per_hour', 'capital_gains', 'capital_losses',
                'dividends_from_stocks', 'num_persons_worked_for_employer',
                'weeks_worked_in_year'
            ],
            'outcome_name': 'income'
        }
        d = DataLoader(params)
        inp_shape = len(d.encoded_feature_names)
        pred_model = BlackBox(inp_shape)
        bs = bs or 32
        lr = lr or 0.01
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        train_path = os.path.join(base_data_dir, 'census', f"{dataset_name}-train-set.npy")
        val_path   = os.path.join(base_data_dir, 'census', f"{dataset_name}-val-set.npy")

    elif dataset_name == 'law':
        dataset = pd.read_csv(os.path.join(base_data_dir, 'law', 'bar_pass_prediction_v2.csv'))
        params = {
            'dataframe': dataset.copy(),
            'continuous_features': ['decile1b','decile3','lsat','ugpa','zfygpa','zgpa'],
            'outcome_name': 'pass_bar'
        }
        d = DataLoader(params)
        inp_shape = len(d.encoded_feature_names)
        pred_model = BlackBox(inp_shape)
        bs = bs or 32
        lr = lr or 0.01
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        train_path = os.path.join(base_data_dir, 'law', f"{dataset_name}-train-set.npy")
        val_path   = os.path.join(base_data_dir, 'law', f"{dataset_name}-val-set.npy")

    elif dataset_name == 'folktables_adult':
        dataset = pd.read_csv(os.path.join(base_data_dir, 'folktables_adult', 'full_data_prep.csv'))
        dataset = dataset.drop(columns=['POBP','RELP'])
        params = {
            'dataframe': dataset.copy(),
            'continuous_features': ['age','hours_per_week'],
            'outcome_name': 'income'
        }
        d = DataLoader(params)
        inp_shape = len(d.encoded_feature_names)
        pred_model = BlackBox(inp_shape)
        bs = bs or 64
        lr = lr or 0.0001
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, pred_model.predict_net.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        train_path = os.path.join(base_data_dir, 'folktables_adult', f"{dataset_name}-train-set.npy")
        val_path   = os.path.join(base_data_dir, 'folktables_adult', f"{dataset_name}-val-set.npy")

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # — Load and shuffle data —
    train_dataset = np.load(train_path)
    print("Train set shape:", train_dataset.shape)
    np.random.shuffle(train_dataset)

    validation_dataset = np.load(val_path)
    print("Validation set shape:", validation_dataset.shape)

    # Balanced sampling for census
    if dataset_name == 'census':
        train_1 = train_dataset[train_dataset[:, -1] == 1]
        train_0 = train_dataset[train_dataset[:, -1] == 0]
        min_len = min(train_0.shape[0], int(0.6 * train_1.shape[0]))
        train_0 = train_0[:min_len]
        train_dataset = np.concatenate((train_1, train_0), axis=0)
        np.random.shuffle(train_dataset)

    # — Training loop —
    for epoch in range(epochs):
        np.random.shuffle(train_dataset)
        batches = np.array_split(train_dataset, len(train_dataset)//bs, axis=0)
        train_acc = 0
        for batch in batches:
            optimizer.zero_grad()
            x = torch.tensor(batch[:, :-1]).float()
            y = torch.tensor(batch[:, -1], dtype=torch.int64)
            outputs = pred_model(x)
            train_acc += torch.sum(torch.argmax(outputs, dim=1) == y).item()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} – Train Acc: {train_acc}/{len(train_dataset)}")

    # — Validation —
    np.random.shuffle(validation_dataset)
    val_batches = np.array_split(validation_dataset, len(validation_dataset)//bs, axis=0)
    val_acc = 0
    for batch in val_batches:
        x = torch.tensor(batch[:, :-1]).float()
        y = torch.tensor(batch[:, -1], dtype=torch.int64)
        outputs = pred_model(x)
        val_acc += torch.sum(torch.argmax(outputs, dim=1) == y).item()
    print(f"Validation Acc: {val_acc}/{len(validation_dataset)}")

    # — Save model —
    os.makedirs(base_model_dir, exist_ok=True)
    save_path = os.path.join(base_model_dir, f"{dataset_name}.pth")
    torch.save(pred_model.state_dict(), save_path)
    print("Model saved to", save_path)

    return pred_model


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python blackbox‑model‑train.py <dataset_name>")
        sys.exit(1)
    train_blackbox(sys.argv[1])
