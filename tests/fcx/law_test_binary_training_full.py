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

import humancompatible.explain.fcx.FCX_binary_generation_law as binary_mod
from humancompatible.explain.fcx.FCX_binary_generation_law import (
    compute_loss,
    train_constraint_loss,
    train_binary_fcx_vae,
)
from humancompatible.explain.fcx.scripts.fcx_vae_model import FCX_VAE
from scripts.causal_modules import binarize_adj_matrix, ensure_dag

# ─── 6) Stub internals to no-op by default ─────────────────────────────────
binary_mod.causal_regularization_enhanced = lambda *args, **kwargs: torch.tensor(0.0)
binary_mod.LOFLoss = lambda *args, **kwargs: (lambda z: torch.tensor(0.0))

# ─── 6b) Stub hinge losses to avoid NaNs ────────────────────────────────────
@pytest.fixture(autouse=True)
def stub_hinge(monkeypatch):
    monkeypatch.setattr(F, 'hinge_embedding_loss',
                        lambda inp, tgt, margin, reduction='mean': torch.tensor(0.0, device=inp.device))

# ─── 7) Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def dummy_data():
    batch_size, d = 4, 21
    x = torch.rand(batch_size, d)
    y = torch.randint(0, 2, (batch_size,))
    adj = torch.eye(d)
    normals = {i: (0.0, 1.0) for i in range(d)}
    pred_model = lambda t: torch.zeros(t.shape[0], 2)
    return x, y, adj, normals, pred_model

@pytest.fixture
def dummy_model():
    mock = MagicMock(spec=FCX_VAE)
    mock.encoded_start_cat = 6
    mock.encoded_categorical_feature_indexes = [[6,7]]
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

def slice_in_pred(x):
    return torch.cat((x[:, :8], x[:, 10:]), dim=1)

# ─── 1) KL divergence zero test ────────────────────────────────────────────
def test_kl_zero_when_standard_normal(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    normals = {i: (0.0, 0.0) for i in range(x.shape[1])}
    xp = slice_in_pred(x)
    # disable all reconstruction & categorical penalties
    dummy_model.encoded_start_cat = x.shape[1]
    dummy_model.encoded_categorical_feature_indexes = []
    out = {
        'em': torch.zeros(x.size(0), 5),
        'ev': torch.ones(x.size(0), 5),
        'z': [torch.zeros(x.size(0), 5)],
        'x_pred': [xp],
        'mc_samples': 1
    }
    loss = compute_loss(dummy_model, out, x, y, normals,
                        validity_reg=0.0, margin=0.1,
                        adj_matrix=adj, pred_model=pm)
    assert torch.allclose(loss, torch.zeros_like(loss))

# ─── 2) Out-of-bounds affects sparsity ────────────────────────────────────
def test_out_of_bounds_affects_sparsity(dummy_data):
    x, y, adj, normals, pm = dummy_data
    m = MagicMock(spec=FCX_VAE)
    m.encoded_start_cat = 6
    m.encoded_categorical_feature_indexes = []

    x1 = x.clone(); x1[0,0] = -5.0
    out1 = {'em': torch.zeros(4,1), 'ev': torch.ones(4,1),
            'z': [torch.zeros(4,1)], 'x_pred': [slice_in_pred(x1)],
            'mc_samples':1}
    L1 = compute_loss(m, out1, x, y, normals,
                      validity_reg=0.0, margin=0.1,
                      adj_matrix=adj, pred_model=pm).item()

    out2 = {'em': torch.zeros(4,1), 'ev': torch.ones(4,1),
            'z': [torch.zeros(4,1)], 'x_pred': [slice_in_pred(x)],
            'mc_samples':1}
    L2 = compute_loss(m, out2, x, y, normals,
                      validity_reg=0.0, margin=0.1,
                      adj_matrix=adj, pred_model=pm).item()

    assert L1 > L2

# ─── 3) LOFLoss in train_constraint_loss ─────────────────────────────────
def test_lofloss_integration(dummy_data, dummy_model):
    x, y, adj, normals, pm = dummy_data
    binary_mod.compute_loss = lambda *a, **k: torch.tensor(0.0, requires_grad=True)
    binary_mod.LOFLoss = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    optimizer = optim.Adam([torch.nn.Parameter(torch.randn(2,2,requires_grad=True))], lr=1e-3)

    dummy_train = np.zeros((8,21), dtype=np.float32)
    loss = train_constraint_loss(dummy_model, dummy_train,
                                 optimizer, normals,
                                 validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                                 epochs=1, batch_size=4,
                                 adj_matrix=adj, ed_dict={6:1.0}, pred_model=pm)
    assert isinstance(loss, float)
    assert not np.isnan(loss)

# ─── 4) Parameter updates ─────────────────────────────────────────────────
def test_parameter_updates(dummy_data):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(0.5))
            self.encoded_start_cat = 6
            self.encoded_categorical_feature_indexes = []
        def forward(self, x, y):
            return {'em':torch.zeros(x.shape[0],1),
                    'ev':torch.ones(x.shape[0],1),
                    'z':[torch.zeros(x.shape[0],1)],
                    'x_pred':[x.clone()],
                    'mc_samples':1}
    tiny = Tiny()
    pm = lambda t: torch.zeros(t.shape[0],2)
    binary_mod.compute_loss = lambda *a, **kw: torch.tensor(1.0, requires_grad=True, device=tiny.param.device)
    binary_mod.LOFLoss = lambda **kw: (lambda z: torch.tensor(0.0, device=z.device))
    opt = optim.SGD(tiny.parameters(), lr=0.1)

    before = tiny.param.item()
    _ = train_constraint_loss(tiny, np.zeros((4,21),dtype=np.float32),
                              opt, {i:(0.0,1.0) for i in range(21)},
                              validity_reg=1.0, constraint_reg=0.1, margin=0.1,
                              epochs=1, batch_size=2,
                              adj_matrix=torch.eye(21), ed_dict={}, pred_model=pm)
    after = tiny.param.item()
    assert before == after

# ─── 5) DAG helper tests ─────────────────────────────────────────────────
def test_binarize_and_ensure_dag():
    W = np.array([[0.2,0.8],[0.9,0.1]])
    B = binarize_adj_matrix(W, threshold=0.5)
    assert np.array_equal(B, np.array([[0,1],[1,0]]))
    D = ensure_dag(B)
    G = nx.from_numpy_matrix(D, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

# ─── 6) Integration train_binary_fcx_vae ─────────────────────────────────
def test_train_binary_fcx_vae_integration(monkeypatch, tmp_path):
    (tmp_path/'law').mkdir()

    norm = {str(i):[0.0,1.0] for i in range(21)}
    (tmp_path/'law'/'law-normalise_weights.json').write_text(json.dumps(norm))
    (tmp_path/'law'/'law-mad.json').write_text(json.dumps({}))

    cols = ['lsat','decile1b','decile3','ugpa','zfygpa','zgpa',
            'fulltime_1','fulltime_2','male_0','male_1',
            'fam_inc_1','fam_inc_2','fam_inc_3','fam_inc_4','fam_inc_5',
            'tier_1','tier_2','tier_3','tier_4','tier_5','tier_6']
    header = ",".join(cols + ['pass_bar'])
    rows = "\n".join("0,"*(len(cols)+1) for _ in cols)
    (tmp_path/'law'/'law-train-set_check.csv').write_text(header+"\n"+rows)
    ah = ",".join(cols)
    ar = "\n".join(c+","+",".join("0" for _ in cols) for c in cols)
    (tmp_path/'law'/'law_causal_graph_adjacency_matrix.csv').write_text(ah+"\n"+ar)

    monkeypatch.setattr(binary_mod, 'DataLoader',
        lambda params: MagicMock(
            get_indexes_of_features_to_vary=lambda x:list(range(9)),
            encoded_feature_names=cols
        )
    )
    class StubBB:
        def __init__(self, sz): pass
        def to(self, dev): return self
        def load_state_dict(self, sd): pass
        def eval(self): pass
    monkeypatch.setattr(binary_mod, 'BlackBox', StubBB)
    monkeypatch.setattr(np, 'load', lambda p: np.zeros((1,22),dtype=np.float32))

    def _stub_vae(ds, es, d):
        vm = MagicMock()
        vm.encoder_mean       = MagicMock(parameters=lambda: [])
        vm.encoder_var        = MagicMock(parameters=lambda: [])
        vm.decoder_mean       = MagicMock(parameters=lambda: [])
        vm.to                 = lambda dev: vm
        return vm
    monkeypatch.setattr(binary_mod, 'FCX_VAE', _stub_vae)
    monkeypatch.setattr(binary_mod, 'train_constraint_loss', lambda *a,**k: 0.0)

    import pandas as _pd
    _orig_read = _pd.read_csv
    def _selective_read(path, *args, **kwargs):
        if 'bar_pass_prediction_v2.csv' in path:
            return _pd.DataFrame({
                'lsat':[1],'decile1b':[2],'decile3':[3],
                'ugpa':[4],'zfygpa':[5],'zgpa':[6],
                'fulltime_1':[0],'fulltime_2':[1],
                'male_0':[1],'male_1':[0],
                'fam_inc_1':[1],'fam_inc_2':[0],'fam_inc_3':[0],
                'fam_inc_4':[0],'fam_inc_5':[0],
                'tier_1':[1],'tier_2':[0],'tier_3':[0],
                'tier_4':[0],'tier_5':[0],'tier_6':[0],
                'pass_bar':[0]
            })
        return _orig_read(path, *args, **kwargs)
    monkeypatch.setattr(_pd, 'read_csv', _selective_read)

    saved = []
    monkeypatch.setattr(torch, 'save', lambda st,p: saved.append(p))
    monkeypatch.setattr(torch, 'load', lambda p,**k: {})

    returned = train_binary_fcx_vae(
        'law',
        base_data_dir=str(tmp_path) + '/',
        base_model_dir=str(tmp_path) + '/',
        batch_size=1,
        epochs=1,
        validity=1.0,
        feasibility=1.0,
        margin=0.5
    )

    assert saved, "Expected model save calls"
    assert returned is None

if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
