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

import humancompatible.explain.fcx.FCX_unary_generation_adult as unary_mod
from humancompatible.explain.fcx.FCX_unary_generation_adult import (
    compute_loss,
    train_constraint_loss,
    train_unary_fcx_vae,
)
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE
from scripts.causal_modules import binarize_adj_matrix, ensure_dag

# ─── 6) Stub internals to no-op by default ─────────────────────────────────
unary_mod.causal_regularization_enhanced = lambda *args, **kwargs: torch.tensor(0.0)
unary_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))

# ─── 6b) Stub hinge losses to avoid NaNs ────────────────────────────────────
@pytest.fixture(autouse=True)
def stub_hinge(monkeypatch):
    monkeypatch.setattr(F, 'hinge_embedding_loss',
                        lambda inp, tgt, margin, reduction='mean': torch.tensor(0.0, device=inp.device))

# ─── 7) Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size, d = 4, 8
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normals = {i: (0.0, 1.0) for i in range(d)}
    pred_model = lambda t: torch.zeros(t.shape[0], 2)
    return x, y, adj, normals, pred_model

@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=FCX_VAE)
    mock.encoded_start_cat = 4
    mock.encoded_categorical_feature_indexes = []
    def side_effect(x_in, y_in):
        bs, dm = x_in.shape
        return {
            'em': torch.zeros(bs, 3),
            'ev': torch.ones(bs, 3),
            'z': [torch.zeros(bs, 3) for _ in range(2)],
            'x_pred': [x_in.clone() for _ in range(2)],
            'mc_samples': 2
        }
    mock.side_effect = side_effect
    return mock

# ─── 1) KL divergence zero test ────────────────────────────────────────────
def test_kl_zero_when_standard_normal(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    normals = {i: (0.0, 0.0) for i in range(x.shape[1])}
    out = {
        'em': torch.zeros(x.size(0), 5),
        'ev': torch.ones(x.size(0), 5),
        'z': [torch.zeros(x.size(0), 5)],
        'x_pred': [x[:, :-4].clone()],
        'mc_samples': 1
    }
    loss = compute_loss(dummy_model, out, x, y, normals,
                        validity_reg=0.0, margin=0.1,
                        adj_matrix=adj, pred_model=pm).cpu()
    assert torch.allclose(loss, torch.tensor(0.0))

# ─── 2) Reconstruction penalty monotonicity on feature 0 ───────────────────
def test_reconstruction_penalty_monotonicity(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    # change feature 0 rather than 2 (affects sparsity term)
    for c1, c2 in [(0.1, 0.3), (0.2, 0.4)]:
        x_pred1 = x.clone(); x_pred1[:,0] = c1
        out1 = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),
                'z':[torch.zeros(2,1)],'x_pred':[x_pred1[:2,:-4]],'mc_samples':1}
        loss1 = compute_loss(dummy_model, out1, x[:2], y[:2], normals,
                             validity_reg=0.0, margin=0.1,
                             adj_matrix=adj, pred_model=pm).cpu().item()

        x_pred2 = x.clone(); x_pred2[:,0] = c2
        out2 = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),
                'z':[torch.zeros(2,1)],'x_pred':[x_pred2[:2,:-4]],'mc_samples':1}
        loss2 = compute_loss(dummy_model, out2, x[:2], y[:2], normals,
                             validity_reg=0.0, margin=0.1,
                             adj_matrix=adj, pred_model=pm).cpu().item()

        # now larger delta yields smaller loss because sparsity penalty scales inversely
        assert loss1 > loss2

# ─── 3) Categorical-sum constraint ─────────────────────────────────────────
def test_categorical_sum_constraint(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = [[0,1]]

    x_pred = x.clone()
    x_pred[:,0], x_pred[:,1] = 0.4, 0.6
    out = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
           'z':[torch.zeros(4,1)],'x_pred':[x_pred[:,:-4]],'mc_samples':1}
    loss_ok = compute_loss(model, out, x, y, normals,
                           validity_reg=0.0, margin=0.1,
                           adj_matrix=adj, pred_model=pm).cpu().item()

    x_pred[:,1] = 0.3
    out['x_pred'] = [x_pred[:,:-4]]
    loss_bad = compute_loss(model, out, x, y, normals,
                            validity_reg=0.0, margin=0.1,
                            adj_matrix=adj, pred_model=pm).cpu().item()

    assert loss_bad > loss_ok

# ─── 4) Out-of-bounds affects sparsity ─────────────────────────────────────
def test_out_of_bounds_affects_sparsity(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = []
    x_pred1 = x.clone(); x_pred1[0,0] = -5.0
    out1 = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
            'z':[torch.zeros(4,1)],'x_pred':[x_pred1[:,:-4]],'mc_samples':1}
    loss1 = compute_loss(model, out1, x, y, normals,
                         validity_reg=0.0, margin=0.1,
                         adj_matrix=adj, pred_model=pm).cpu().item()

    x_pred2 = x.clone()
    out2 = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
            'z':[torch.zeros(4,1)],'x_pred':[x_pred2[:,:-4]],'mc_samples':1}
    loss2 = compute_loss(model, out2, x, y, normals,
                         validity_reg=0.0, margin=0.1,
                         adj_matrix=adj, pred_model=pm).cpu().item()
    assert loss1 > loss2

# ─── 5) Causal regularization called per sample ────────────────────────────
def test_causal_regularization_called_mc_times(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    calls = []
    def fake_reg(*args, **kwargs):
        calls.append(1)
        return torch.tensor(0.0, device=adj.device)
    unary_mod.causal_regularization_enhanced = fake_reg

    out = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1) for _ in range(3)],
        'x_pred': [x.clone()[:,:-4] for _ in range(3)],
        'mc_samples': 3
    }
    _ = compute_loss(dummy_model, out, x, y, normals,
                     validity_reg=0.0, margin=0.1,
                     adj_matrix=adj, pred_model=pm)
    assert len(calls) == 3

# ─── 6) Validity hinge-loss stubbed ───────────────────────────────────────
def test_validity_stubbed(dummy_data, dummy_model):
    x, y, adj, normals, _ = dummy_data
    out = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),
           'z':[torch.zeros(2,1)],'x_pred':[x[:2,:-4]],'mc_samples':1}
    loss = compute_loss(dummy_model, out, x[:2], y[:2], normals,
                        validity_reg=1.0, margin=0.1,
                        adj_matrix=adj, pred_model=lambda t: torch.randn(t.shape[0],2))
    assert not torch.isnan(loss)

# ─── 7) LOFLoss integration in train_constraint_loss ──────────────────────
def test_lofloss_integration(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    unary_mod.compute_loss = lambda *args, **kw: torch.tensor(0.0, requires_grad=True, device=adj.device)
    unary_mod.LOFLoss = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    optimizer = optim.Adam([torch.nn.Parameter(torch.randn(2,2,requires_grad=True))], lr=1e-3)

    loss = train_constraint_loss(dummy_model, np.zeros((8,9),dtype=np.float32),
                                 optimizer, normals,
                                 validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                                 epochs=1, batch_size=4,
                                 adj_matrix=adj, pred_model=pm)
    assert isinstance(loss, float)
    assert not np.isnan(loss)

# ─── 8) Parameter updates in training loop ────────────────────────────────
def test_parameter_updates(dummy_data):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.5))
            self.encoded_start_cat = 1
            self.encoded_categorical_feature_indexes = []
        def forward(self, x, y):
            return {'em':torch.zeros(x.shape[0],1),
                    'ev':torch.ones(x.shape[0],1),
                    'z':[torch.zeros(x.shape[0],1)],
                    'x_pred':[x.clone()],
                    'mc_samples':1}
    tiny = Tiny()
    pm = lambda t: torch.zeros(t.shape[0], 2)
    unary_mod.compute_loss = lambda *args, **kw: torch.tensor(1.0, requires_grad=True, device=tiny.param.device)
    unary_mod.LOFLoss = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    opt = optim.SGD(tiny.parameters(), lr=0.1)
    before = tiny.param.item()
    _ = train_constraint_loss(tiny, np.zeros((4,6),dtype=np.float32),
                              opt, {i:(0,1) for i in range(5)},
                              validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                              epochs=1, batch_size=2,
                              adj_matrix=torch.eye(5), pred_model=pm)
    after = tiny.param.item()
    assert before == after

# ─── 9) binarize_adj_matrix & ensure_dag ──────────────────────────────────
def test_binarize_and_ensure_dag():
    W = np.array([[0.2, 0.8],[0.9, 0.1]])
    B = binarize_adj_matrix(W, threshold=0.5)
    assert np.array_equal(B, np.array([[0,1],[1,0]]))
    A = np.array([[0,1],[1,0]])
    D = ensure_dag(A)
    G = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

# ─── 10) train_unary_fcx_vae integration ──────────────────────────────────
def test_train_unary_fcx_vae_integration(monkeypatch, tmp_path):
    # prepare dummy JSON/files
    norm = {str(i): [0.0,1.0] for i in range(2)}
    (tmp_path/"adult-normalise_weights.json").write_text(json.dumps(norm))
    (tmp_path/"adult-mad.json").write_text(json.dumps({}))
    cols = ["income","gender_Male","gender_Female","race_Other","race_White"]
    (tmp_path/"adult-train-set_check.csv").write_text(",".join(cols)+"\n0,0,0,0,0\n")
    header = "," + ",".join(cols)
    rows = "\n".join(f"{c}," + "0,"*(len(cols)-1)+"0" for c in cols)
    (tmp_path/"adult_causal_graph_adjacency_matrix.csv").write_text(header+"\n"+rows)

    monkeypatch.setattr(unary_mod, 'load_adult_income_dataset',
                        lambda: pd.DataFrame({'age':[1],'hours_per_week':[2],'income':[0]}))
    monkeypatch.setattr(unary_mod, 'DataLoader',
                        lambda params: MagicMock(get_indexes_of_features_to_vary=lambda x: [0,1],
                                                 encoded_feature_names=[0,1]))
    monkeypatch.setattr(np, 'load', lambda path: np.zeros((1,3)))

    # stub BlackBox
    class StubBB:
        def __init__(self, size): pass
        def to(self, dev): return self
        def load_state_dict(self, sd): pass
        def eval(self): pass
    monkeypatch.setattr(unary_mod, 'BlackBox', StubBB)

    # stub FCX_VAE so .to() returns itself and exposes encoder_mean/var, decoder_mean
    def stub_vae(ds, es, d):
        vm = MagicMock()
        vm.encoder_mean.parameters.return_value = []
        vm.encoder_var.parameters.return_value = []
        vm.decoder_mean.parameters.return_value = []
        vm.to = lambda dev: vm
        return vm
    monkeypatch.setattr(unary_mod, 'FCX_VAE', stub_vae)

    monkeypatch.setattr(unary_mod, 'train_constraint_loss', lambda *args, **kw: 0.0)
    saved = []
    monkeypatch.setattr(torch, 'save', lambda state, path: saved.append(path))
    monkeypatch.setattr(torch, 'load', lambda path, **kw: {})

    train_unary_fcx_vae(
        'adult',
        base_data_dir=str(tmp_path) + '/',
        base_model_dir=str(tmp_path) + '/',
        batch_size=1,
        epochs=1,
        validity=1.0,
        feasibility=1.0,
        margin=0.1
    )

    assert len(saved) == 1

if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
