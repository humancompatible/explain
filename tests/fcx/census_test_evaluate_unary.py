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
import humancompatible.explain.fcx.evaluate_unary_census as eval_mod
from humancompatible.explain.fcx.evaluate_unary_census import evaluate_census

# ─── 6) Monkey‑patch all external dependencies ─────────────────────────────
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    # stub DataLoader → simple loader
    class DummyLoader:
        def __init__(self, params):
            # encoded_feature_names length only matters for size
            self.encoded_feature_names = [f'f{i}' for i in range(9)]
        def get_indexes_of_features_to_vary(self, feats):
            return list(range(len(self.encoded_feature_names)))
    monkeypatch.setattr(eval_mod, 'DataLoader', DummyLoader)

    # stub pd.read_csv to return a dummy DataFrame for both census_data and adjacency
    dummy_df = pd.DataFrame({
        'age': [0],
        'wage_per_hour': [0],
        'capital_gains': [0],
        'capital_losses': [0],
        'dividends_from_stocks': [0],
        'num_persons_worked_for_employer': [0],
        'weeks_worked_in_year': [0],
        'income': [0]
    })
    monkeypatch.setattr(eval_mod.pd, 'read_csv', lambda *args, **kwargs: dummy_df)

    # stub np.load to return toy test‐set (with trailing label 0)
    arr = np.hstack([np.zeros((5, 9)), np.zeros((5,1))])
    monkeypatch.setattr(np, 'load', lambda path: arr)

    # stub builtins.open + json.load for normalise_weights and mad
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setattr('builtins.open', lambda path, mode='r': DummyFile())

    normalise = {str(i): [0.0,1.0] for i in range(9)}
    mad       = {str(i): 0.1 for i in range(9)}
    seq = {'first': True}
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
        def __call__(self, x): return torch.zeros(len(x), 2)
    monkeypatch.setattr(eval_mod, 'BlackBox', BB)

    # stub compute_eval_metrics to simply echo metric_idx
    def fake_eval(immutables, methods, base_model_dir, encoded_size,
                  pred_model, vae_test_dataset, loader,
                  normalise_weights, mad_feature_weights,
                  div_case, metric_idx, sample_range,
                  metric_name, prefix_name=None):
        return float(metric_idx)
    monkeypatch.setattr(eval_mod, 'compute_eval_metrics', fake_eval)

# ─── 7) Test evaluate_census ─────────────────────────────────────────────────
def test_evaluate_census_unary_mapping():
    out = evaluate_census(
        base_data_dir='any/',
        base_model_dir='any/',
        dataset_name='census',
        pth_name='dummy.pth'
    )
    # Should contain only one top‐level key: 'census'
    assert set(out.keys()) == {'census'}
    metrics = out['census']

    expected_keys = {'validity','const-score','cont-prox','cat-prox','LOF'}
    assert set(metrics.keys()) == expected_keys

    # Since fake_eval returns its metric_idx:
    assert metrics['validity']    == 0.0   # first call uses idx=0
    assert metrics['const-score'] == 1.0   # second call uses idx=1
    assert metrics['cont-prox']   == 3.0   # third call uses idx=3
    assert metrics['cat-prox']    == 4.0   # fourth call uses idx=4
    assert metrics['LOF']         == 8.0   # fifth call uses idx=8

# ─── 8) Allow direct execution ───────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
