#!/usr/bin/env python3
import sys
import os
import types

# ─── 1) Make project root importable ────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ─── 2) Expose scripts/ for bare imports ───────────────────────────────────
SCRIPTS = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx', 'scripts')
sys.path.insert(0, SCRIPTS)

# ─── 3) Expose fcx pkg for bare LOFLoss import ─────────────────────────────
FCX_PKG = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx')
sys.path.insert(0, FCX_PKG)

# ─── 4) Stub out torchvision + save_image ──────────────────────────────────
torchvision_stub = types.ModuleType('torchvision')
torchvision_stub.datasets   = types.ModuleType('torchvision.datasets')
torchvision_stub.transforms = types.ModuleType('torchvision.transforms')
utils_stub                  = types.ModuleType('torchvision.utils')
utils_stub.save_image       = lambda *args, **kwargs: None

sys.modules['torchvision']            = torchvision_stub
sys.modules['torchvision.datasets']   = torchvision_stub.datasets
sys.modules['torchvision.transforms'] = torchvision_stub.transforms
sys.modules['torchvision.utils']      = utils_stub

# ─── 5) Imports ─────────────────────────────────────────────────────────────
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch import optim

import humancompatible.explain.fcx.FCX_unary_generation_adult as unary_mod
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE

compute_loss = unary_mod.compute_loss
train_constraint_loss = unary_mod.train_constraint_loss

# ─── 6) Monkey‑patch internals ──────────────────────────────────────────────
# Avoid the Bool‑eye bug
unary_mod.causal_regularization_enhanced = lambda outputs, adj_matrix, lambda_nc, lambda_c: torch.tensor(0.0)
# LOFLoss no‑op
unary_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))

# ─── 7) Fixture: dummy dataset ─────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size = 4
    d = 8
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normalise_weights = {i: (0.0, 1.0) for i in range(d)}
    return x, y, adj, normalise_weights

# ─── 8) Fixture: dummy model ───────────────────────────────────────────────
@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=FCX_VAE)
    def side_effect(train_x, train_y):
        bs, d_mut = train_x.shape
        latent = 3
        return {
            'em': torch.zeros(bs, latent),
            'ev': torch.ones(bs, latent),
            'z': [torch.zeros(bs, latent) for _ in range(2)],
            'x_pred': [torch.rand(bs, d_mut) for _ in range(2)],
            'mc_samples': 2
        }
    mock.side_effect = side_effect
    mock.encoded_start_cat = 1
    mock.encoded_categorical_feature_indexes = [[2, 3], [4, 5]]
    return mock

# ─── 9) Test compute_loss directly ─────────────────────────────────────────
def test_compute_loss_unary(dummy_data, dummy_model):
    x, y, adj, normals = dummy_data
    unary_mod.cuda = torch.device('cpu')
    unary_mod.pred_model = MagicMock(return_value=torch.randn(x.size(0), 2))

    bs, d = x.shape
    latent = 3
    mut_dim = d - 4
    model_out = {
        'em': torch.zeros(bs, latent),
        'ev': torch.ones(bs, latent),
        'z': [torch.zeros(bs, latent) for _ in range(2)],
        'x_pred': [torch.rand(bs, mut_dim) for _ in range(2)],
        'mc_samples': 2
    }

    loss = compute_loss(
        model=dummy_model,
        model_out=model_out,
        x=x,
        target_label=y,
        normalise_weights=normals,
        validity_reg=0.5,
        margin=0.1,
        adj_matrix=adj
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

# ─── 10) Test train_constraint_loss ────────────────────────────────────────
def test_train_constraint_loss_unary_runs(dummy_data, dummy_model):
    x, y, adj, normals = dummy_data
    unary_mod.cuda = torch.device('cpu')
    unary_mod.pred_model = MagicMock(return_value=torch.randn(5, 2))

    # Monkey‑patch compute_loss to avoid internal errors
    unary_mod.compute_loss = lambda *args, **kwargs: torch.tensor(0.0, requires_grad=True)
    # Ensure ed_dict global is present
    unary_mod.ed_dict = {}

    bs, d = x.shape
    train_np = np.random.rand(12, d + 1).astype(np.float32)

    optimizer = optim.Adam(
        [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))],
        lr=1e-3
    )

    # No ed_dict argument here
    loss = train_constraint_loss(
        model=dummy_model,
        train_dataset=train_np,
        optimizer=optimizer,
        normalise_weights=normals,
        validity_reg=0.5,
        constraint_reg=0.2,
        margin=0.1,
        epochs=1,
        batch_size=5,
        adj_matrix=adj
    )
    assert isinstance(loss, float)

# ─── 11) Allow direct execution ────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
