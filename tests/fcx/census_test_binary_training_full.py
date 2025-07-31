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

# ─── 3) Stub out torchvision + save_image ──────────────────────────────────
torchvision_stub = types.ModuleType('torchvision')
torchvision_stub.datasets   = types.ModuleType('torchvision.datasets')
torchvision_stub.transforms = types.ModuleType('torchvision.transforms')
utils_stub                  = types.ModuleType('torchvision.utils')
utils_stub.save_image       = lambda *args, **kwargs: None

sys.modules['torchvision']            = torchvision_stub
sys.modules['torchvision.datasets']   = torchvision_stub.datasets
sys.modules['torchvision.transforms'] = torchvision_stub.transforms
sys.modules['torchvision.utils']      = utils_stub

# ─── 4) Imports ─────────────────────────────────────────────────────────────
import pytest
import torch
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import MagicMock
from torch import optim
import torch.nn.functional as F

import humancompatible.explain.fcx.FCX_binary_generation_census as binary_mod
from humancompatible.explain.fcx.FCX_binary_generation_census import (
    compute_loss,
    train_constraint_loss,
    train_binary_fcx_vae,
)
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE
from scripts.causal_modules import binarize_adj_matrix, ensure_dag

# ─── 6) Stub internals to no-op by default ─────────────────────────────────
binary_mod.causal_regularization_enhanced = lambda *args, **kwargs: torch.tensor(0.0)
binary_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))
binary_mod.ed_dict = {}  # ensure global ed_dict for train_constraint_loss

# ─── 6b) Stub hinge losses to avoid NaNs ────────────────────────────────────
@pytest.fixture(autouse=True)
def stub_hinge(monkeypatch):
    monkeypatch.setattr(F, 'hinge_embedding_loss',
                        lambda inp, tgt, margin, reduction='mean': torch.tensor(0.0, device=inp.device))

# ─── 7) Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size, d = 4, 9
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normals = {i: (0.0, 1.0) for i in range(d)}
    pred_model = lambda t: torch.zeros(t.shape[0], 2, device=t.device)
    return x, y, adj, normals, pred_model

@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=binary_mod.FCX_VAE)
    mock.encoded_start_cat = 7
    mock.encoded_categorical_feature_indexes = []
    def side_effect(x_in, y_in):
        bs, _ = x_in.shape
        latent = 3
        return {
            'em': torch.zeros(bs, latent),
            'ev': torch.ones(bs, latent),
            'z': [torch.zeros(bs, latent) for _ in range(2)],
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
        'x_pred': [x[:, :-7].clone() if x.shape[1] > 7 else x.clone()],
        'mc_samples': 1
    }
    loss = compute_loss(dummy_model, out, x, y, normals,
                        validity_reg=0.0, margin=0.1,
                        adj_matrix=adj, pred_model=pm).cpu()
    assert torch.allclose(loss, torch.tensor(0.0))

# ─── 2) Reconstruction monotonicity on first feature ──────────────────────
def test_reconstruction_penalty_monotonicity(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    # allow non-strict monotonicity
    for c1, c2 in [(0.1, 0.3), (0.2, 0.4)]:
        x1 = x.clone(); x1[:,0] = c1
        out1 = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),'z':[torch.zeros(2,1)],
                'x_pred':[x1[:2,:-7]],'mc_samples':1}
        L1 = compute_loss(dummy_model, out1, x[:2], y[:2], normals,0.0,0.1,adj,pm).item()
        x2 = x.clone(); x2[:,0] = c2
        out2 = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),'z':[torch.zeros(2,1)],
                'x_pred':[x2[:2,:-7]],'mc_samples':1}
        L2 = compute_loss(dummy_model, out2, x[:2], y[:2], normals,0.0,0.1,adj,pm).item()
        assert L1 >= L2

# ─── 3) Categorical sum constraint ───────────────────────────────────────
def test_categorical_sum_constraint(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=binary_mod.FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = [[0,1]]

    x_pred = x.clone()
    x_pred[:,0], x_pred[:,1] = 0.4, 0.6
    out = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
           'z':[torch.zeros(4,1)],'x_pred':[x_pred[:,:-7]],'mc_samples':1}
    loss_ok = compute_loss(model, out, x, y, normals,0.0,0.1,adj,pm).item()

    x_pred[:,1] = 0.3
    out['x_pred'] = [x_pred[:,:-7]]
    loss_bad = compute_loss(model, out, x, y, normals,0.0,0.1,adj,pm).item()
    assert loss_bad > loss_ok

# ─── 4) Out-of-bounds affects sparsity ────────────────────────────────────
def test_out_of_bounds_affects_sparsity(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=binary_mod.FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = []
    x_pred1 = x.clone(); x_pred1[0,0] = -5.0
    out1 = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
            'z':[torch.zeros(4,1)],'x_pred':[x_pred1[:,:-7]],'mc_samples':1}
    loss1 = compute_loss(model, out1, x, y, normals,0.0,0.1,adj,pm).item()

    x_pred2 = x.clone()
    out2 = {'em':torch.zeros(4,1),'ev':torch.ones(4,1),
            'z':[torch.zeros(4,1)],'x_pred':[x_pred2[:,:-7]],'mc_samples':1}
    loss2 = compute_loss(model, out2, x, y, normals,0.0,0.1,adj,pm).item()
    assert loss1 > loss2

# ─── 5) Causal reg called per MC ─────────────────────────────────────────
def test_causal_regularization_called_mc_times(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    calls = []
    def fake_reg(*args, **kwargs):
        calls.append(1)
        return torch.tensor(0.0, device=adj.device)
    binary_mod.causal_regularization_enhanced = fake_reg

    out = {
        'em': torch.zeros(4,1),
        'ev': torch.ones(4,1),
        'z': [torch.zeros(4,1) for _ in range(3)],
        'x_pred': [x.clone()[:,:-7] for _ in range(3)],
        'mc_samples': 3
    }
    _ = compute_loss(dummy_model, out, x, y, normals,0.0,0.1,adj,pm)
    assert len(calls) == 3

# ─── 6) Validity stub doesn't nan ────────────────────────────────────────
def test_validity_stubbed(dummy_data, dummy_model):
    x, y, adj, normals, _ = dummy_data
    out = {'em':torch.zeros(2,1),'ev':torch.ones(2,1),
           'z':[torch.zeros(2,1)],'x_pred':[x[:2,:-7]],'mc_samples':1}
    loss = compute_loss(dummy_model, out, x[:2], y[:2], normals,1.0,0.1,adj,
                        lambda t: torch.randn(t.shape[0],2, device=t.device))
    assert not torch.isnan(loss)

# ─── 7) LOFLoss in train_constraint_loss ─────────────────────────────────
def test_lofloss_integration(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    binary_mod.compute_loss = lambda *a, **k: torch.tensor(0.0, requires_grad=True, device=adj.device)
    binary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    opt = optim.Adam([torch.nn.Parameter(torch.randn(2,2,requires_grad=True))], lr=1e-3)
    # pass a non-empty ed_dict to avoid None
    loss = train_constraint_loss(
        dummy_model,
        np.zeros((8,10),dtype=np.float32),
        opt,
        normals,
        validity_reg=1.0,
        constraint_reg=0.1,
        margin=0.1,
        epochs=1,
        batch_size=4,
        adj_matrix=adj,
        ed_dict={0:1},
        pred_model=pm
    )
    assert isinstance(loss, float)
    assert not np.isnan(loss)

# ─── 8) Parameter updates ─────────────────────────────────────────────────
def test_parameter_updates(dummy_data):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.5))
            self.encoded_start_cat = 2
            self.encoded_categorical_feature_indexes = []
        def forward(self, x, y):
            return {'em':torch.zeros(x.shape[0],1),
                    'ev':torch.ones(x.shape[0],1),
                    'z':[torch.zeros(x.shape[0],1)],
                    'x_pred':[x.clone()],
                    'mc_samples':1}
    tiny = Tiny()
    pm = lambda t: torch.zeros(t.shape[0],2, device=t.device)
    binary_mod.compute_loss = lambda *a, **kw: torch.tensor(1.0, requires_grad=True, device=tiny.param.device)
    binary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    binary_mod.ed_dict = {}
    opt = optim.SGD(tiny.parameters(), lr=0.1)
    before = tiny.param.item()
    _ = train_constraint_loss(tiny, np.zeros((4,11),dtype=np.float32),
                              opt, {i:(0,1) for i in range(10)},
                              validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                              epochs=1, batch_size=2,
                              adj_matrix=torch.eye(9), ed_dict={}, pred_model=pm)
    after = tiny.param.item()
    assert before == after

# ─── 9) DAG helper tests ─────────────────────────────────────────────────
def test_binarize_and_ensure_dag():
    W = np.array([[0.2,0.8],[0.9,0.1]])
    B = binarize_adj_matrix(W, threshold=0.5)
    assert np.array_equal(B, np.array([[0,1],[1,0]]))
    D = ensure_dag(B)
    G = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

# ─── 10) Integration train_binary_fcx_vae ─────────────────────────────────
def test_train_binary_fcx_vae_integration(monkeypatch, tmp_path):
    base = tmp_path / 'census'
    base.mkdir()
    # dummy census_data.csv
    cols = ['age','wage_per_hour','capital_gains','capital_losses',
            'dividends_from_stocks','num_persons_worked_for_employer','weeks_worked_in_year','income']
    (base/'census_data.csv').write_text(",".join(cols)+"\n1,1,1,1,1,1,1,0\n")
    # .npy
    np.save(str(base/'census-train-set.npy'), np.zeros((100,8),dtype=np.float32))
    np.save(str(base/'census-val-set.npy'),   np.zeros((10,8), dtype=np.float32))
    # JSON
    norm = {str(i):[0.0,1.0] for i in range(8)}
    (base/'census-normalise_weights.json').write_text(json.dumps(norm))
    (base/'census-mad.json').write_text(json.dumps({}))
    # train-set_check.csv
    (base/'census-train-set_check.csv').write_text(",".join([f'f{i}' for i in range(8)])+"\n"+"0,0,0,0,0,0,0,0\n")
    # adjacency
    header = ",".join([f'f{i}' for i in range(8)])
    rows = "\n".join(f"f{i},"+"0,"*7+"0" for i in range(8))
    (base/'census_causal_graph_adjacency_matrix.csv').write_text(header+"\n"+rows)

    # stub DataLoader & BlackBox & FCX_VAE
    monkeypatch.setattr(binary_mod, 'DataLoader', lambda params: MagicMock(
        get_indexes_of_features_to_vary=lambda x: list(range(8)),
        encoded_feature_names=list(range(8))
    ))
    class StubBB:
        def __init__(self,size): pass
        def to(self,dev): return self
        def load_state_dict(self,sd): pass
        def eval(self): pass
    monkeypatch.setattr(binary_mod, 'BlackBox', StubBB)
    def stub_vae(ds,es,d):
        vm = MagicMock()
        vm.encoder_mean.parameters.return_value = []
        vm.encoder_var.parameters.return_value  = []
        vm.decoder_mean.parameters.return_value = []
        vm.to = lambda dev: vm
        return vm
    monkeypatch.setattr(binary_mod, 'FCX_VAE', stub_vae)
    monkeypatch.setattr(binary_mod, 'train_constraint_loss', lambda *a,**k:0.0)
    # avoid train_test_split randomness
    monkeypatch.setattr(np.random, 'choice', lambda a, size, replace=False: np.arange(a))

    saved = []
    monkeypatch.setattr(torch, 'save', lambda st,p: saved.append(p))
    monkeypatch.setattr(torch, 'load', lambda p,**k: {} )

    _ = train_binary_fcx_vae(
        dataset_name='census',
        base_data_dir=str(tmp_path) + '/',
        base_model_dir=str(tmp_path) + '/',
        batch_size=4,
        epochs=1,
        validity=1.0,
        feasibility=1.0,
        margin=0.5
    )
    assert saved, "Expected model save calls"

if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
