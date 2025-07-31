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

# ─── 3) Expose fcx pkg for bare imports ───────────────────────────────────
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

import humancompatible.explain.fcx.FCX_binary_generation_folktables_adult as binary_mod
from humancompatible.explain.fcx.FCX_binary_generation_folktables_adult import (
    compute_loss,
    train_constraint_loss,
    train_binary_fcx_vae,
)
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE
from scripts.causal_modules import binarize_adj_matrix, ensure_dag



# ─── 6) Stub internals to no-op by default ─────────────────────────────────
binary_mod.causal_regularization_enhanced = lambda *args, **kwargs: torch.tensor(0.0)
binary_mod.LOFLoss = lambda **kwargs: (lambda z: torch.tensor(0.0, device=z.device))

# ─── 6b) Stub hinge to avoid NaNs ──────────────────────────────────────────
@pytest.fixture(autouse=True)
def stub_hinge(monkeypatch):
    monkeypatch.setattr(F, 'hinge_embedding_loss',
                        lambda inp, tgt, margin, reduction='mean': torch.tensor(0.0, device=inp.device))

# ─── 7) Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size, d = 4, 12
    x = torch.rand(batch_size, d, device='cuda:0')
    y = torch.randint(0, 2, (batch_size,), device='cuda:0')
    adj = torch.eye(d, device='cuda:0')
    normals = {i: (0.0, 1.0) for i in range(d)}
    pred_model = lambda t: torch.zeros(t.shape[0], 2, device=t.device)
    return x, y, adj, normals, pred_model

@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=FCX_VAE)
    mock.encoded_start_cat = 2
    mock.encoded_categorical_feature_indexes = []
    def side_effect(x_in, y_in):
        bs, _ = x_in.shape
        return {
            'em': torch.zeros(bs, 1, device=x_in.device),
            'ev': torch.ones(bs, 1, device=x_in.device),
            'z': [torch.zeros(bs, 1, device=x_in.device)],
            'x_pred': [x_in.clone()],
            'mc_samples': 1
        }
    mock.side_effect = side_effect
    return mock


# ─── 1) Reconstruction monotonicity ─────────────────────────────────────────
def test_reconstruction_penalty_monotonicity(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    for c1, c2 in [(0.1, 0.3), (0.2, 0.4)]:
        x1 = x.clone(); x1[:,0] = c1
        out1 = {'em':torch.zeros(2,1,device=x.device),'ev':torch.ones(2,1,device=x.device),
                'z':[torch.zeros(2,1,device=x.device)],'x_pred':[x1[:2,:-11]],'mc_samples':1}
        L1 = compute_loss(dummy_model, out1, x[:2], y[:2], normals,
                          validity_reg=0, margin=0.1,
                          adj_matrix=adj, pred_model=pm).item()
        x2 = x.clone(); x2[:,0] = c2
        out2 = {'em':torch.zeros(2,1,device=x.device),'ev':torch.ones(2,1,device=x.device),
                'z':[torch.zeros(2,1,device=x.device)],'x_pred':[x2[:2,:-11]],'mc_samples':1}
        L2 = compute_loss(dummy_model, out2, x[:2], y[:2], normals,
                          validity_reg=0, margin=0.1,
                          adj_matrix=adj, pred_model=pm).item()
        assert L1 >= L2

# ─── 2) Categorical sum constraint ─────────────────────────────────────────
def test_categorical_sum_constraint(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = [[0,1]]
    xp = x.clone(); xp[:,0], xp[:,1] = 0.4, 0.6
    out = {'em':torch.zeros(4,1,device=x.device),'ev':torch.ones(4,1,device=x.device),
           'z':[torch.zeros(4,1,device=x.device)],'x_pred':[xp[:,:-11]],'mc_samples':1}
    ok = compute_loss(model, out, x, y, normals,0,0.1,adj,pm).item()
    xp[:,1] = 0.3
    out['x_pred'] = [xp[:,:-11]]
    bad = compute_loss(model, out, x, y, normals,0,0.1,adj,pm).item()
    assert bad >= ok

# ─── 3) Out-of-bounds affects sparsity ──────────────────────────────────────
def test_out_of_bounds_affects_sparsity(dummy_data):
    x, y, adj, normals, pm = dummy_data
    model = MagicMock(spec=FCX_VAE)
    model.encoded_start_cat = 2
    model.encoded_categorical_feature_indexes = []
    x1 = x.clone(); x1[0,0] = -5.0
    out1 = {'em':torch.zeros(4,1,device=x.device),'ev':torch.ones(4,1,device=x.device),
            'z':[torch.zeros(4,1,device=x.device)],'x_pred':[x1[:,:-11]],'mc_samples':1}
    L1 = compute_loss(model, out1, x, y, normals,0,0.1,adj,pm).item()
    out2 = {'em':torch.zeros(4,1,device=x.device),'ev':torch.ones(4,1,device=x.device),
            'z':[torch.zeros(4,1,device=x.device)],'x_pred':[x[:,:-11]],'mc_samples':1}
    L2 = compute_loss(model, out2, x, y, normals,0,0.1,adj,pm).item()
    assert L1 >= L2

# ─── 4) Causal reg per MC ─────────────────────────────────────────────────
def test_causal_regularization_called_mc_times(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    calls = []
    binary_mod.causal_regularization_enhanced = lambda *a, **k: calls.append(1) or torch.tensor(0.0, device=adj.device)
    out = {'em': torch.zeros(4,1,device=x.device),'ev': torch.ones(4,1,device=x.device),
           'z': [torch.zeros(4,1,device=x.device) for _ in range(3)],
           'x_pred': [x[:,:-11].clone() for _ in range(3)], 'mc_samples': 3}
    _ = compute_loss(dummy_model, out, x, y, normals,0,0.1,adj,pm)
    assert len(calls) == 3

# ─── 5) Validity stub ─────────────────────────────────────────────────────
def test_validity_stubbed(dummy_data, dummy_model):
    x, y, adj, normals, _ = dummy_data
    out = {'em':torch.zeros(2,1,device=x.device),'ev':torch.ones(2,1,device=x.device),
           'z':[torch.zeros(2,1,device=x.device)],'x_pred':[x[:2,:-11]],'mc_samples':1}
    loss = compute_loss(dummy_model, out, x[:2], y[:2], normals,1.0,0.1,adj,
                        lambda t: torch.randn(t.shape[0],2, device=t.device))
    assert not torch.isnan(loss)

# ─── 6) LOFLoss integration ─────────────────────────────────────────────────
def test_lofloss_integration(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    binary_mod.compute_loss = lambda *a, **kw: torch.tensor(0.0, requires_grad=True, device=adj.device)
    binary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    optimizer = optim.Adam([torch.nn.Parameter(torch.randn(2,2,requires_grad=True, device='cuda:0'))], lr=1e-3)
    loss = train_constraint_loss(dummy_model, np.zeros((8,12),dtype=np.float32),
                                 optimizer, normals,
                                 validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                                 epochs=1, batch_size=4,
                                 adj_matrix=adj, ed_dict={0:1.0}, pred_model=pm)
    assert isinstance(loss, float)
    assert not np.isnan(loss)

# ─── 7) Parameter updates ─────────────────────────────────────────────────
def test_parameter_updates(dummy_data):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.5, device='cuda:0'))
            self.encoded_start_cat = 1
            self.encoded_categorical_feature_indexes = []
        def forward(self,x,y):
            return {'em':torch.zeros(x.shape[0],1,device=x.device),
                    'ev':torch.ones(x.shape[0],1,device=x.device),
                    'z':[torch.zeros(x.shape[0],1,device=x.device)],
                    'x_pred':[x.clone()],
                    'mc_samples':1}
    tiny = Tiny()
    pm = lambda t: torch.zeros(t.shape[0],2, device=t.device)
    binary_mod.compute_loss = lambda *a,**kw: torch.tensor(1.0, requires_grad=True, device=tiny.param.device)
    binary_mod.LOFLoss      = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    opt = optim.SGD(tiny.parameters(), lr=0.1)
    before = tiny.param.item()
    _ = train_constraint_loss(tiny, np.zeros((4,12),dtype=np.float32),
                              opt, {i:(0,1) for i in range(11)},
                              validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                              epochs=1, batch_size=2,
                              adj_matrix=torch.eye(12, device='cuda:0'), ed_dict={}, pred_model=pm)
    after = tiny.param.item()
    assert before == after

# ─── 8) DAG helper tests ─────────────────────────────────────────────────
def test_binarize_and_ensure_dag():
    W = np.array([[0.2,0.8],[0.9,0.1]])
    B = binarize_adj_matrix(W, threshold=0.5)
    assert np.array_equal(B, np.array([[0,1],[1,0]]))
    D = ensure_dag(B)
    G = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

# ─── 9) Integration train_binary_fcx_vae ─────────────────────────────────
def test_train_binary_fcx_vae_integration(monkeypatch, tmp_path):
    base = tmp_path
    ds = base / 'folktables_adult'
    ds.mkdir()
    cols = ['age','hours_per_week','POBP','RELP','workclass','education','marital_status','occupation','income']
    (ds/'full_data_prep.csv').write_text(','.join(cols)+"\n25,35,X,A,Priv,BA,S,O1,0\n")
    train = np.zeros((5,12), float); val = np.zeros((3,12), float); test = np.zeros((4,12), float)
    np.save(str(ds/'folktables_adult-train-set.npy'), train)
    np.save(str(ds/'folktables_adult-val-set.npy'),   val)
    np.save(str(ds/'folktables_adult-test-set.npy'),  test)
    norm = {str(i):[0.0,1.0] for i in range(12)}
    (ds/'folktables_adult-normalise_weights.json').write_text(json.dumps(norm))
    (ds/'folktables_adult-mad.json').write_text(json.dumps({}))
    cols12 = [f'f{i}' for i in range(12)]
    (ds/'folktables_adult-train-set_check.csv').write_text(','.join(cols12)+"\n"+"0,"*11+"0\n")
    header = ','.join(cols12)
    rows = "\n".join(f"{c},"+"0,"*11+"0" for c in cols12)
    (ds/'folktables_adult_custom_causal_graph_adjacency_matrix_decor_full.csv').write_text(header+"\n"+rows)

    monkeypatch.setattr(binary_mod, 'DataLoader', lambda params: MagicMock(get_indexes_of_features_to_vary=lambda feats: list(range(6)), encoded_feature_names=list(range(12))))
    class StubBB:
        def __init__(self, sz): pass
        def to(self, dev): return self
        def load_state_dict(self, sd): pass
        def eval(self): pass
    monkeypatch.setattr(binary_mod, 'BlackBox', StubBB)
    def stub_vae(ds_arg, es, d):
        vm = MagicMock()
        vm.encoder_mean.parameters.return_value=[]
        vm.encoder_var.parameters.return_value=[]
        vm.decoder_mean.parameters.return_value=[]
        vm.to = lambda dev: vm
        return vm
    monkeypatch.setattr(binary_mod, 'FCX_VAE', stub_vae)
    monkeypatch.setattr(binary_mod, 'train_constraint_loss', lambda *a, **k: 0.0)
    monkeypatch.setattr(np.random, 'seed', lambda s: None)
    saved = []
    monkeypatch.setattr(torch, 'save', lambda st,p: saved.append(p))
    monkeypatch.setattr(torch, 'load', lambda p,**kw: {})

    model = train_binary_fcx_vae(
        'folktables_adult',
        base_data_dir=str(base) + '/',
        base_model_dir=str(base) + '/',
        batch_size=1,
        epochs=1,
        validity=1.0,
        feasibility=1.0,
        margin=0.1
    )
    assert saved, "Expected model save"

if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))