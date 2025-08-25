#!/usr/bin/env python3
import sys, os, types

# ─── 1) Project root on PYTHONPATH ─────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ─── 2) Expose scripts/ for bare imports ──────────────────────────────────
SCRIPTS = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx', 'scripts')
sys.path.insert(0, SCRIPTS)

# ─── 3) Expose fcx pkg for any bare LOFLoss imports ────────────────────────
FCX_PKG = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx')
sys.path.insert(0, FCX_PKG)

# ─── 4) Stub out torchvision + save_image ─────────────────────────────────
torchvision_stub = types.ModuleType('torchvision')
torchvision_stub.datasets   = types.ModuleType('torchvision.datasets')
torchvision_stub.transforms = types.ModuleType('torchvision.transforms')
utils_stub                  = types.ModuleType('torchvision.utils')
utils_stub.save_image       = lambda *args, **kwargs: None
sys.modules['torchvision']            = torchvision_stub
sys.modules['torchvision.datasets']   = torchvision_stub.datasets
sys.modules['torchvision.transforms'] = torchvision_stub.transforms
sys.modules['torchvision.utils']      = utils_stub

# ─── 5) Imports ────────────────────────────────────────────────────────────
import pytest
import torch
import numpy as np
import pandas as pd
import json

# import the binary evaluation script from scripts/
import humancompatible.explain.fcx.evaluate_binary_adult as eval_mod
from humancompatible.explain.fcx.evaluate_binary_adult import evaluate_adult

# ─── 6) Monkey‑patch external dependencies ─────────────────────────────────
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    # 6.1) Stub load_adult_income_dataset → small DataFrame
    df = pd.DataFrame({
        'age': [20, 30, 40],
        'hours_per_week': [35,40,45],
        'income': [0,0,1],
    })
    monkeypatch.setattr(eval_mod, 'load_adult_income_dataset', lambda: df)

    # 6.2) DataLoader stub
    class DL:
        def __init__(self, params):
            self.encoded_feature_names = ['age','hours_per_week']
        def get_indexes_of_features_to_vary(self, feats):
            return [0,1]
    monkeypatch.setattr(eval_mod, 'DataLoader', DL)

    # 6.3) np.load → toy array (4 rows, last col labels)
    arr = np.array([[1,2,0],[3,4,0],[5,6,1],[7,8,0]], float)
    monkeypatch.setattr(np, 'load', lambda path: arr)

    # 6.4) JSON loads for normalise and mad
    normalise = {'0':[0.0,1.0],'1':[0.0,1.0]}
    mad       = {'age':0.1,'hours_per_week':0.2}
    # patch builtins.open
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return ''
    monkeypatch.setattr('builtins.open', lambda path, mode='r': DummyFile())
    # patch json.load to alternate between normalise and mad
    seq = {'first':True}
    def fake_json_load(fp):
        if seq.pop('first', False):
            return normalise
        return mad
    monkeypatch.setattr(json, 'load', fake_json_load)

    # 6.5) torch.load stub
    monkeypatch.setattr(torch, 'load', lambda p: {})

    # 6.6) BlackBox stub
    class BB:
        def __init__(self, sz): pass
        def load_state_dict(self, d): pass
        def eval(self): pass
        def __call__(self, x): return torch.zeros(len(x),2)
    monkeypatch.setattr(eval_mod, 'BlackBox', BB)

    # 6.7) FCX_VAE stub
    class VAE:
        def __init__(self,*a,**k): pass
        def load_state_dict(self, d): pass
        def eval(self): pass
    monkeypatch.setattr(eval_mod, 'FCX_VAE', VAE)

    # 6.8) compute_eval_metrics_adult stub: returns metric_idx
    def fake_eval(immut, methods, bmd, enc_sz, pred, test_ds, loader,
                  norm_wt, mad_wt, div_case, metric_idx, sample_range,
                  metric_name, prefix_name=None):
        return float(metric_idx)
    monkeypatch.setattr(eval_mod, 'compute_eval_metrics_adult', fake_eval)

# ─── 7) Test evaluate_adult for binary ──────────────────────────────────────
def test_evaluate_binary_metrics():
    out = evaluate_adult(
        base_data_dir='whatever/',
        base_model_dir='whatever/',
        dataset_name='adult',
        pth_name='binary.pth'
    )
    # Should have top‐level key 'adult'
    assert set(out.keys()) == {'adult'}
    m = out['adult']
    expected = {'validity','const-score','cont-prox','cat-prox','LOF'}
    assert set(m.keys()) == expected
    # check their values = the respective metric indices
    assert m['validity']    == 0.0
    assert m['const-score'] == 2.0
    assert m['cont-prox']   == 3.0
    assert m['cat-prox']    == 4.0
    assert m['LOF']         == 7.0

# ─── 8) Direct exec ────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
