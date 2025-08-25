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

# import the census‐evaluation module from scripts/
import humancompatible.explain.fcx.evaluate_binary_folktables_adult as eval_mod
from humancompatible.explain.fcx.evaluate_binary_folktables_adult import evaluate_folktables_adult


# ─── 6) Monkey‑patch external dependencies ─────────────────────────────────
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    # 6.1) Stub pd.read_csv so that `full_data_prep.csv` loads and drop works
    dummy_df = pd.DataFrame({
        'age': [25, 35, 45],
        'hours_per_week': [30, 40, 50],
        'POBP': ['X','Y','Z'],
        'RELP': ['A','B','C'],
        'income': [0, 0, 1],
    })
    monkeypatch.setattr(eval_mod.pd, 'read_csv', lambda *args, **kwargs: dummy_df.copy())

    # 6.2) DataLoader stub
    class DL:
        def __init__(self, params):
            # After drop, loader only needs encoded_feature_names
            self.encoded_feature_names = ['age','hours_per_week']
        def get_indexes_of_features_to_vary(self, feats):
            return [0,1]
    monkeypatch.setattr(eval_mod, 'DataLoader', DL)

    # 6.3) np.load → toy test‐set with extra label column
    arr = np.hstack([np.zeros((4,2)), np.zeros((4,1))])
    monkeypatch.setattr(np, 'load', lambda path: arr)

    # 6.4) open/json.load for normalise_weights and mad
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setattr('builtins.open', lambda path, mode='r': DummyFile())

    normalise = {'0':[0.0,1.0],'1':[0.0,1.0]}
    mad       = {'age':0.1,'hours_per_week':0.2}
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

    # 6.7) compute_eval_metrics_adult stub: returns metric_idx
    def fake_eval(immut, methods, bmd, enc_sz, pred, test_ds, loader,
                  norm_wt, mad_wt, div_case, metric_idx, sample_range,
                  metric_name, prefix_name=None):
        return float(metric_idx)
    monkeypatch.setattr(eval_mod, 'compute_eval_metrics_adult', fake_eval)

# ─── 7) Test evaluate_folktables_adult ─────────────────────────────────────
def test_evaluate_folktables_adult_mapping():
    out = evaluate_folktables_adult(
        base_data_dir='any/',
        base_model_dir='any/',
        dataset_name='folktables_adult',
        pth_name='dummy.pth'
    )
    # top‐level key must match dataset_name
    assert set(out.keys()) == {'folktables_adult'}
    metrics = out['folktables_adult']

    expected_keys = {'validity','const-score','cont-prox','cat-prox','LOF'}
    assert set(metrics.keys()) == expected_keys

    # fake_eval returns its metric_idx:
    # calls at idx 0,2,3,4,7
    assert metrics['validity']    == 0.0
    assert metrics['const-score'] == 2.0
    assert metrics['cont-prox']   == 3.0
    assert metrics['cat-prox']    == 4.0
    assert metrics['LOF']         == 7.0

# ─── 8) Direct exec ────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))