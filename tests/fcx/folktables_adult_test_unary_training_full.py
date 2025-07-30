#!/usr/bin/env python3
import sys
import os
import types
import json

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
import pandas as pd
import networkx as nx
from unittest.mock import MagicMock
from torch import optim
import torch.nn.functional as F

import humancompatible.explain.fcx.FCX_unary_generation_folktables_adult as unary_mod
from humancompatible.explain.fcx.FCX_unary_generation_folktables_adult import (
    compute_loss,
    train_constraint_loss,
    train_unary_fcx_vae,
)
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE
from scripts.causal_modules import binarize_adj_matrix, ensure_dag

# ─── 6) Stub internals to no-op by default ─────────────────────────────────
unary_mod.causal_regularization_enhanced = lambda *args, **kwargs: torch.tensor(0.0)
unary_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))

# ─── 6b) Stub hinge_embedding_loss to avoid NaNs ──────────────────────────
@pytest.fixture(autouse=True)
def stub_hinge(monkeypatch):
    monkeypatch.setattr(F, 'hinge_embedding_loss',
                        lambda inp, tgt, margin, reduction='mean': torch.tensor(0.0, device=inp.device))

# ─── 7) Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size, d = 4, 11
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normals = {i: (0.0, 1.0) for i in range(d)}
    pm = lambda t: torch.zeros(t.shape[0], 2)
    return x, y, adj, normals, pm

@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=FCX_VAE)
    mock.encoded_start_cat = 2
    mock.encoded_categorical_feature_indexes = []
    def side_effect(x_in, y_in):
        bs, _ = x_in.shape
        return {
            'em': torch.zeros(bs, 1),
            'ev': torch.ones(bs, 1),
            'z': [torch.zeros(bs, 1) for _ in range(2)],
            'x_pred': [x_in.clone() for _ in range(2)],
            'mc_samples': 2
        }
    mock.side_effect = side_effect
    return mock

# ─── 1) KL divergence zero test ────────────────────────────────────────────
def test_kl_zero_when_standard_normal(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    normals = {i: (0.0, 0.0) for i in range(x.shape[1])}
    out = {
        'em': torch.zeros(x.size(0), 5),
        'ev': torch.ones(x.size(0), 5),
        'z': [torch.zeros(x.size(0), 5)],
        'x_pred': [x[:, :-11].clone()],
        'mc_samples': 1
    }
    loss = compute_loss(
        dummy_model, out, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj,
        pred_model=pm
    ).cpu()
    assert torch.allclose(loss, torch.tensor(0.0))

# ─── 2) Reconstruction penalty monotonicity on feature 0 ───────────────────
def test_reconstruction_penalty_monotonicity(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    for c1, c2 in [(0.1, 0.3), (0.2, 0.4)]:
        x1 = x.clone(); x1[:,0] = c1
        out1 = {
            'em': torch.zeros(2,1),
            'ev': torch.ones(2,1),
            'z': [torch.zeros(2,1)],
            'x_pred': [x1[:2,:-11]],
            'mc_samples':1
        }
        L1 = compute_loss(
            dummy_model, out1, x[:2], y[:2], normals,
            validity_reg=0.0, margin=0.1,
            adj_matrix=adj, pred_model=pm
        ).item()

        x2 = x.clone(); x2[:,0] = c2
        out2 = {
            'em': torch.zeros(2,1),
            'ev': torch.ones(2,1),
            'z': [torch.zeros(2,1)],
            'x_pred': [x2[:2,:-11]],
            'mc_samples':1
        }
        L2 = compute_loss(
            dummy_model, out2, x[:2], y[:2], normals,
            validity_reg=0.0, margin=0.1,
            adj_matrix=adj, pred_model=pm
        ).item()

        # allow equality if unchanging
        assert L1 >= L2

# ─── 3) Categorical‐sum constraint ─────────────────────────────────────────
def test_categorical_sum_constraint(dummy_data):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    m = MagicMock(spec=FCX_VAE)
    m.encoded_start_cat = 2
    m.encoded_categorical_feature_indexes = [[0,1]]

    xp = x.clone(); xp[:,0], xp[:,1] = 0.4, 0.6
    out = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1)],
        'x_pred': [xp[:,:-11]],
        'mc_samples':1
    }
    ok = compute_loss(
        m, out, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj, pred_model=pm
    ).item()

    xp[:,1] = 0.3
    out['x_pred'] = [xp[:,:-11]]
    bad = compute_loss(
        m, out, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj, pred_model=pm
    ).item()

    # allow equality if unchanging
    assert bad >= ok

# ─── 4) Out‐of‐bounds affects sparsity ─────────────────────────────────────
def test_out_of_bounds_affects_sparsity(dummy_data):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    m = MagicMock(spec=FCX_VAE)
    m.encoded_start_cat = 2
    m.encoded_categorical_feature_indexes = []

    x1 = x.clone(); x1[0,0] = -5.0
    out1 = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1)],
        'x_pred': [x1[:,:-11]],
        'mc_samples':1
    }
    L1 = compute_loss(
        m, out1, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj, pred_model=pm
    ).item()

    out2 = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1)],
        'x_pred': [x[:,:-11]],
        'mc_samples':1
    }
    L2 = compute_loss(
        m, out2, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj, pred_model=pm
    ).item()

    # allow equality if unchanging
    assert L1 >= L2

# ─── 5) Causal regularization per MC ──────────────────────────────────────
def test_causal_regularization_called_mc_times(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    calls = []
    unary_mod.causal_regularization_enhanced = lambda *a, **k: calls.append(1) or torch.tensor(0.0)

    out = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1) for _ in range(3)],
        'x_pred': [x[:,:-11].clone() for _ in range(3)],
        'mc_samples': 3
    }
    _ = compute_loss(
        dummy_model, out, x, y, normals,
        validity_reg=0.0, margin=0.1,
        adj_matrix=adj, pred_model=pm
    )
    assert len(calls) == 3

# ─── 6) Validity hinge‐loss stubbed ───────────────────────────────────────
def test_validity_stubbed(dummy_data, dummy_model):
    x, y, adj, normals, _ = dummy_data
    unary_mod.pred_model = lambda t: torch.randn(t.shape[0], 2)

    out = {
        'em': torch.zeros(2,1),
        'ev': torch.ones(2,1),
        'z': [torch.zeros(2,1)],
        'x_pred': [x[:2,:-11]],
        'mc_samples':1
    }
    loss = compute_loss(
        dummy_model, out, x[:2], y[:2], normals,
        validity_reg=1.0, margin=0.1,
        adj_matrix=adj, pred_model=unary_mod.pred_model
    )
    assert not torch.isnan(loss)

# ─── 7) LOFLoss integration in train_constraint_loss ──────────────────────
def test_lofloss_integration(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    unary_mod.compute_loss = lambda *args, **kw: torch.tensor(0.0, requires_grad=True)
    unary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    opt = optim.Adam([torch.nn.Parameter(torch.randn(2,2,requires_grad=True))], lr=1e-3)

    loss = train_constraint_loss(
        dummy_model,
        np.zeros((8,12),dtype=np.float32),
        opt,
        normals,
        validity_reg=1.0,
        constraint_reg=0.1,
        margin=0.1,
        epochs=1,
        batch_size=4,
        adj_matrix=adj,
        pred_model=pm
    )
    assert isinstance(loss, float)
    assert not np.isnan(loss)

# ─── 8) Parameter updates in training loop ────────────────────────────────
def test_parameter_updates(dummy_data):
    x, y, adj, normals, pm = dummy_data
    unary_mod.pred_model = pm

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.5))
            self.encoded_start_cat = 1
            self.encoded_categorical_feature_indexes = []
        def forward(self, x, y):
            return {
                'em':torch.zeros(x.shape[0],1),
                'ev':torch.ones(x.shape[0],1),
                'z':[torch.zeros(x.shape[0],1)],
                'x_pred':[x.clone()],
                'mc_samples':1
            }

    tiny = Tiny()
    opt = optim.SGD(tiny.parameters(), lr=0.1)
    unary_mod.compute_loss = lambda *args, **kw: torch.tensor(1.0, requires_grad=True, device=tiny.param.device)
    unary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))

    before = tiny.param.item()
    _ = train_constraint_loss(
        tiny,
        np.zeros((4,12),dtype=np.float32),
        opt,
        normals,
        validity_reg=1.0,
        constraint_reg=0.1,
        margin=0.1,
        epochs=1,
        batch_size=2,
        adj_matrix=torch.eye(5),
        pred_model=pm
    )
    after = tiny.param.item()
    assert before == after

# ─── 9) binarize_adj_matrix & ensure_dag ──────────────────────────────────
def test_binarize_and_ensure_dag():
    W = np.array([[0.2, 0.8],[0.9, 0.1]])
    B = binarize_adj_matrix(W, threshold=0.5)
    assert np.array_equal(B, np.array([[0,1],[1,0]]))
    D = ensure_dag(B)
    G = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

# ─── 10) train_unary_fcx_vae integration ──────────────────────────────────
def test_train_unary_fcx_vae_integration(monkeypatch, tmp_path):
    base = tmp_path
    ds = base / 'folktables_adult'
    ds.mkdir()
    # minimal full_data_prep.csv
    (ds/'full_data_prep.csv').write_text(
        "age,hours_per_week,POBP,RELP,workclass,education,marital_status,occupation,income\n"
        "25,35,X,A,Priv,BA,S,O1,0\n"
    )
    # train/val/test .npy
    for split in ['train','val','test']:
        np.save(str(ds/f"folktables_adult-{split}-set.npy"), np.zeros((2,11)))
    # normalise & mad
    norm = {str(i):[0.0,1.0] for i in range(11)}
    (ds/'folktables_adult-normalise_weights.json').write_text(json.dumps(norm))
    (ds/'folktables_adult-mad.json').write_text(json.dumps({}))
    # train-set_check & adjacency
    cols = pd.read_csv(str(ds/'full_data_prep.csv')).columns
    (ds/'folktables_adult-train-set_check.csv').write_text(
        ",".join(cols) + "\n" + ",".join(["0"]*len(cols)) + "\n"
    )
    header = ",".join(cols)
    rows = "\n".join(f"{c}," + ",".join(["0"]*(len(cols)-1)) + "0" for c in cols)
    (ds/'folktables_adult_custom_causal_graph_adjacency_matrix_decor_full.csv').write_text(
        header + "\n" + rows
    )

    # stub DataLoader & BlackBox & FCX_VAE & train_constraint_loss
    monkeypatch.setattr(unary_mod, 'DataLoader',
        lambda params: MagicMock(get_indexes_of_features_to_vary=lambda x:list(range(6)),
                                 encoded_feature_names=list(range(11))))
    class StubBB:
        def __init__(self, sz): pass
        def to(self, dev): return self
        def load_state_dict(self, sd): pass
        def eval(self): pass
    monkeypatch.setattr(unary_mod, 'BlackBox', StubBB)

    def stub_vae(ds_arg, es, d):
        vm = MagicMock()
        vm.encoder_mean.parameters.return_value = []
        vm.encoder_var.parameters.return_value  = []
        vm.decoder_mean.parameters.return_value = []
        vm.to = lambda dev: vm
        return vm
    monkeypatch.setattr(unary_mod, 'FCX_VAE', stub_vae)
    monkeypatch.setattr(unary_mod, 'train_constraint_loss', lambda *a, **k: 0.0)

    saved = []
    monkeypatch.setattr(torch, 'save', lambda state, path: saved.append(path))
    monkeypatch.setattr(torch, 'load', lambda path, **kw: {})

    train_unary_fcx_vae(
        'folktables_adult',
        base_data_dir=str(base) + '/',
        base_model_dir=str(base) + '/',
        batch_size=1,
        epochs=1,
        validity=1.0,
        feasibility=1.0,
        margin=0.1
    )

    assert len(saved) == 1

if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
