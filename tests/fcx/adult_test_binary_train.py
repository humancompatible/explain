#!/usr/bin/env python3
import sys, os, types

# ─── 1) Make project root importable as a package ────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ─── 2) Expose scripts/ for bare imports (dataloader, etc.) ─────────────────
SCRIPTS = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx', 'scripts')
sys.path.insert(0, SCRIPTS)

# ─── 3) Expose the fcx pkg for bare LOFLoss import ─────────────────────────
FCX_PKG = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx')
sys.path.insert(0, FCX_PKG)

# ─── 4) Stub out torchvision + torchvision.utils.save_image ────────────────
torchvision_stub = types.ModuleType('torchvision')
torchvision_stub.datasets   = types.ModuleType('torchvision.datasets')
torchvision_stub.transforms = types.ModuleType('torchvision.transforms')
utils_stub                  = types.ModuleType('torchvision.utils')
utils_stub.save_image       = lambda *args, **kwargs: None

sys.modules['torchvision']            = torchvision_stub
sys.modules['torchvision.datasets']   = torchvision_stub.datasets
sys.modules['torchvision.transforms'] = torchvision_stub.transforms
sys.modules['torchvision.utils']      = utils_stub

# ─── 5) Import pytest + your code under test ────────────────────────────────
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

import humancompatible.explain.fcx.FCX_binary_generation_adult as train_mod
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE

compute_loss = train_mod.compute_loss
train_constraint_loss = train_mod.train_constraint_loss

# ─── 6) Monkey‑patch out complex modules ─────────────────────────────────────
# No-op causal regularization
train_mod.causal_regularization_enhanced = lambda outputs, adj_matrix, lambda_nc, lambda_c: torch.tensor(0.0)
# No-op LOFLoss
train_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))

# ─── 7) Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    batch_size = 4
    d = 10
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normalise_weights = {i: (0.0, 1.0) for i in range(d)}
    return x, y, adj, normalise_weights

@pytest.fixture
def dummy_model(dummy_data):
    x, y, _, _ = dummy_data
    batch, d = x.shape
    latent_dim = 5
    mut_dim = d - 4

    model = MagicMock(spec=FCX_VAE)
    # Prepare model_out for compute_loss
    em = torch.zeros(batch, latent_dim)
    ev = torch.ones(batch, latent_dim)
    z = [torch.zeros(batch, latent_dim) for _ in range(3)]
    raw_preds = [torch.rand(batch, mut_dim) for _ in range(3)]
    model_out = {
        'em': em,
        'ev': ev,
        'z': z,
        'x_pred': raw_preds,
        'mc_samples': 3,
    }
    model.return_value = model_out
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = [[2, 3], [4, 5]]
    return model, model_out

# ─── 8) Tests ────────────────────────────────────────────────────────────────

def test_compute_loss(dummy_model, dummy_data):
    model, model_out = dummy_model
    x, y, adj, norm_wt = dummy_data

    train_mod.cuda = torch.device('cpu')
    train_mod.pred_model = MagicMock(return_value=torch.randn(x.shape[0], 2))

    loss = compute_loss(
        model=model,
        model_out=model_out,
        x=x,
        target_label=y,
        normalise_weights=norm_wt,
        validity_reg=1.0,
        margin=0.5,
        adj_matrix=adj,
    )
    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # now returns a 0-d scalar

def test_train_constraint_loss_runs(dummy_model, dummy_data):
    model, model_out = dummy_model
    x, y, adj, norm_wt = dummy_data
    ed_dict = {2: 1, 4: 2}

    train_mod.cuda = torch.device('cpu')
    train_mod.pred_model = MagicMock(return_value=torch.randn(x.shape[0], 2))

    # ─── fix x_pred shape to match (D+1)-4 ────────────────────────────────────
    D = x.shape[1]
    mut_dim = (D + 1) - 4
    model_out['x_pred'] = [torch.rand(x.shape[0], mut_dim) for _ in range(model_out['mc_samples'])]
    model.return_value = model_out

    N, d = x.shape
    train_np = np.random.rand(16, d + 1).astype(np.float32)

    optimizer = torch.optim.Adam(
        [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))],
        lr=0.001,
    )

    loss = train_constraint_loss(
        model=model,
        train_dataset=train_np,
        optimizer=optimizer,
        normalise_weights=norm_wt,
        validity_reg=1.0,
        constraint_reg=0.1,
        margin=0.5,
        epochs=1,
        batch_size=4,
        adj_matrix=adj,
        ed_dict=ed_dict,
    )
    assert isinstance(loss, float)

# ─── 9) Enable running this file directly ───────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
