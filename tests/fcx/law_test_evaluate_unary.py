#!/usr/bin/env python3
import sys, os, types, json

# ─── 1) Project root on PYTHONPATH ─────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# ─── 2) Expose scripts/ for bare imports ──────────────────────────────────
SCRIPTS = os.path.join(ROOT, 'humancompatible', 'explain', 'fcx', 'scripts')
sys.path.insert(0, SCRIPTS)

# ─── 3) Expose fcx pkg for bare imports ───────────────────────────────────
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

import humancompatible.explain.fcx.evaluate_unary_law as eval_mod
from humancompatible.explain.fcx.evaluate_unary_law import evaluate_law_unary

# ─── 6) Monkey‐patch every external I/O or heavy dependency ───────────────
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch, tmp_path):
    # a) pd.read_csv → three different CSVs
    def fake_read_csv(path, *args, **kwargs):
        fname = os.path.basename(path)
        # bar_pass_prediction_v2.csv → wide df with all columns + pass_bar
        if fname == 'bar_pass_prediction_v2.csv':
            cols = [
                'lsat','decile1b','decile3','ugpa','zfygpa','zgpa',
                'fulltime_1','fulltime_2','male_0','male_1',
                'fam_inc_1','fam_inc_2','fam_inc_3','fam_inc_4','fam_inc_5',
                'tier_1','tier_2','tier_3','tier_4','tier_5','tier_6',
                'pass_bar'
            ]
            # two rows for simplicity
            return pd.DataFrame({c:[0,1] for c in cols})
        # law-train-set_check.csv → same columns, one row of zeros
        if fname == 'law-train-set_check.csv':
            df = fake_read_csv(os.path.join('', 'bar_pass_prediction_v2.csv'))
            df.iloc[:] = 0
            return df
        # law_causal_graph_adjacency_matrix.csv → square zeros
        if fname == 'law_causal_graph_adjacency_matrix.csv':
            df0 = fake_read_csv(os.path.join('', 'bar_pass_prediction_v2.csv'))
            cols = df0.columns
            return pd.DataFrame(0, index=cols, columns=cols)
        # fallback
        return pd.DataFrame()
    monkeypatch.setattr(pd, 'read_csv', fake_read_csv)

    # b) np.load → toy numpy array with 3 test‐rows, last column is label
    #    we want two 0’s and one 1 so the filtering step will leave 2 rows.
    arr = np.array([
        [0]*21 + [0],
        [0]*21 + [1],
        [0]*21 + [0],
    ], dtype=float)
    monkeypatch.setattr(np, 'load', lambda path: arr)

    # c) open()+json.load() → first call returns normalise_weights, second returns mad
    normalise = {str(i):[0.0,1.0] for i in range(21)}
    mad         = {}
    seq = {'first':True}
    class DummyF:
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setattr('builtins.open', lambda *a,**k: DummyF())
    def fake_json_load(fp):
        if seq.pop('first', False):
            return normalise
        return mad
    monkeypatch.setattr(json, 'load', fake_json_load)

    # d) stub DataLoader
    class DL:
        def __init__(self, params):
            # mimic the full encoded_feature_names list
            self.encoded_feature_names = fake_read_csv('bar_pass_prediction_v2.csv').columns[:-1].tolist()
        def get_indexes_of_features_to_vary(self, feats):
            return list(range(9))
    monkeypatch.setattr(eval_mod, 'DataLoader', DL)

    # e) stub BlackBox
    class BB:
        def __init__(self, sz): pass
        def load_state_dict(self, d): pass
        def eval(self): pass
        def __call__(self, x): return torch.zeros(len(x),2)
    monkeypatch.setattr(eval_mod, 'BlackBox', BB)

    # f) stub FCX_VAE
    class VAE:
        def __init__(self,*a,**k): pass
        def to(self, dev): return self
    monkeypatch.setattr(eval_mod, 'FCX_VAE', VAE)

    # g) stub compute_eval_metrics → echo back the metric_idx it was passed
    def fake_eval(immut, methods, bmd, enc_sz, pred, test_ds, loader,
                  norm_wt, mad_wt, div_case, metric_idx, sample_range,
                  metric_name, prefix_name=None):
        return float(metric_idx)
    monkeypatch.setattr(eval_mod, 'compute_eval_metrics', fake_eval)
    
    
    import torch
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})
# ─── 7) Now test evaluate_law_unary ────────────────────────────────────────
def test_evaluate_law_unary(tmp_path):
    # create the law/ subdir (even though everything is stubbed, the code checks existence)
    (tmp_path / 'law').mkdir()

    out = evaluate_law_unary(
        base_data_dir = str(tmp_path) + '/',
        base_model_dir= str(tmp_path) + '/',
        dataset_name  = 'law',
        pth_name      = 'doesn’t-matter.pth'
    )

    # top level key
    assert set(out) == {'law'}
    m = out['law']

    # should produce exactly these five metrics
    assert set(m) == {'validity','const-score','cont-prox','cat-prox','LOF'}

    # and because our fake_eval simply returns its metric_idx,
    # we know evaluate_law_unary calls them in the order 0,1,3,4,8:
    assert m['validity']     == 0.0
    assert m['const-score']  == 1.0
    assert m['cont-prox']    == 3.0
    assert m['cat-prox']     == 4.0
    assert m['LOF']          == 8.0

# ─── 8) boilerplate ───────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.exit(pytest.main(['-q', __file__]))
