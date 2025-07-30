#!/usr/bin/env python3
import sys, os, types

# ─── 1) Make project root importable ────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ─── 2) Expose scripts/ for bare imports ───────────────────────────────────
SCRIPTS = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx', 'scripts')
sys.path.insert(0, SCRIPTS)

# ─── 3) Expose fcx pkg for any bare LOFLoss imports ────────────────────────
FCX_PKG = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx')
sys.path.insert(0, FCX_PKG)

# ─── 4) Stub out torchvision + save_image to avoid PIL issues ──────────────
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
import json

# import the unary‐evaluation module from scripts/
import humancompatible.explain.fcx.evaluate_unary_adult as eval_mod
from humancompatible.explain.fcx.evaluate_unary_adult import evaluate_adult

# ─── 6) Monkey‑patch all external dependencies ─────────────────────────────
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    # stub load_adult_income_dataset → simple DataFrame
    df = pd.DataFrame({
        'age': [25, 35, 45],
        'hours_per_week': [30, 40, 50],
        'income': [0, 0, 1],
    })
    monkeypatch.setattr(eval_mod, 'load_adult_income_dataset', lambda: df)

    # stub DataLoader
    class DummyLoader:
        def __init__(self, params):
            self.encoded_feature_names = ['age','hours_per_week']
        def get_indexes_of_features_to_vary(self, feats):
            return [0,1]
    monkeypatch.setattr(eval_mod, 'DataLoader', DummyLoader)

    # stub np.load to return toy test‐set
    arr = np.array([[1,2,0],[3,4,0],[5,6,1],[7,8,0]], float)
    monkeypatch.setattr(np, 'load', lambda path: arr)

    # stub builtins.open + json.load for normalise_weights and mad
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def read(self): return ''
    monkeypatch.setattr('builtins.open', lambda path, mode='r': DummyFile())

    normalise = {'0':[0.0,1.0],'1':[0.0,1.0]}
    mad       = {'age':0.1,'hours_per_week':0.2}
    seq = {'first':True}
    def fake_json_load(fp):
        if seq.pop('first', False):
            return normalise
        return mad
    monkeypatch.setattr(json, 'load', fake_json_load)

    # stub torch.load
    monkeypatch.setattr(torch, 'load', lambda p: {})

    # stub BlackBox
    class BB:
        def __init__(self, sz): pass
        def load_state_dict(self, d): pass
        def eval(self): pass
        def __call__(self, x): return torch.zeros(len(x),2)
    monkeypatch.setattr(eval_mod, 'BlackBox', BB)

    # stub FCX_VAE (unused here)
    class VAE:
        def __init__(self,*a,**k): pass
        def load_state_dict(self, d): pass
        def eval(self): pass
    monkeypatch.setattr(eval_mod, 'FCX_VAE', VAE)

    # stub compute_eval_metrics_adult to return metric_idx
    def fake_eval(immutables, methods, base_model_dir, encoded_size,
                  pred_model, vae_test_dataset, loader,
                  normalise_weights, mad_feature_weights,
                  div_case, metric_idx, sample_range,
                  metric_name, prefix_name=None):
        return float(metric_idx)
    monkeypatch.setattr(eval_mod, 'compute_eval_metrics_adult', fake_eval)

# ─── 7) Test evaluate_adult ─────────────────────────────────────────────────
def test_evaluate_adult_unary_mapping():
    out = evaluate_adult(
        base_data_dir='any/',
        base_model_dir='any/',
        dataset_name='adult',
        pth_name='dummy.pth'
    )
    # Should contain only one key: 'adult'
    assert set(out.keys()) == {'adult'}
    metrics = out['adult']
    expected_keys = {'validity','const-score','cont-prox','cat-prox','LOF'}
    assert set(metrics.keys()) == expected_keys

    # Since fake_eval returns its metric_idx:
    assert metrics['validity']    == 0.0
    assert metrics['const-score'] == 2.0
    assert metrics['cont-prox']   == 3.0
    assert metrics['cat-prox']    == 4.0
    assert metrics['LOF']         == 7.0

# ─── 8) Allow direct execution ───────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
