import numpy as np
import pandas as pd
import pytest

from humancompatible.explain.lice.data.DataHandler import DataHandler
from humancompatible.explain.lice.data.Features import Binary, Categorical, Contiguous, Mixed, Monotonicity


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 30, 40],
        "gender": ["M", "F", "M"],
        "income": [1000, 2000, 3000],
        "target": [0, 1, 0],
    })


def test_init_with_dataframe(sample_df):
    handler = DataHandler(sample_df, categ_map={"gender":[]}, discrete=["age"], target_name="target")
    assert handler.n_features == 3
    assert handler.target_feature is not None
    assert set(handler.feature_names) == {"age", "gender", "income"}

    handler = DataHandler(sample_df, categ_map={"gender":["M", "F"]}, target_name="target")
    assert handler.n_features == 3
    assert handler.target_feature is not None
    assert set(handler.feature_names) == {"age", "gender", "income"}


def test_init_with_numpy():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    handler = DataHandler(X, y, feature_names=["f1", "f2"])
    assert handler.n_features == 2
    assert handler.target_feature is not None


def test_invalid_feature_names_length():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        DataHandler(X, feature_names=["only_one"])


def test_encode_decode_roundtrip(sample_df):
    handler = DataHandler(sample_df, categ_map={"gender":[]}, target_name="target")
    X = sample_df.drop(columns="target")

    enc = handler.encode(X)
    dec = handler.decode(enc)

    assert isinstance(enc, np.ndarray)
    assert isinstance(dec, pd.DataFrame)
    assert list(dec.columns) == handler.feature_names
    assert dec.shape[0] == X.shape[0]


def test_encode_y_and_decode_y():
    y = np.array(["0", "1", "0", "1"])
    handler = DataHandler(np.array([[1], [2], [3], [4]]), y, feature_names=["f1"])

    enc_y = handler.encode_y(y)
    dec_y = handler.decode_y(enc_y, as_series=False)

    assert isinstance(enc_y, np.ndarray)
    assert len(enc_y) == len(y)
    assert (dec_y == y).all()

    y = np.array([0, 1, 0, 1])
    handler = DataHandler(np.array([[1], [2], [3], [4]]), y, feature_names=["f1"])

    enc_y = handler.encode_y(y)
    dec_y = handler.decode_y(enc_y, as_series=False)

    assert isinstance(enc_y, np.ndarray)
    assert len(enc_y) == len(y)
    assert np.allclose(dec_y.astype(float), y.astype(float))


def test_encode_all(sample_df):
    handler = DataHandler(sample_df, categ_map={"gender":[]}, target_name="target")
    X_all = sample_df.to_numpy()
    enc_all = handler.encode_all(X_all, normalize=True, one_hot=True)

    assert isinstance(enc_all, np.ndarray)
    assert enc_all.shape[0] == sample_df.shape[0]


def test_encoding_width_matches_encode(sample_df):
    handler = DataHandler(sample_df, categ_map={"gender":[]}, target_name="target")
    X = sample_df.drop(columns="target")
    enc = handler.encode(X)

    assert enc.shape[1] == handler.encoding_width(one_hot=True)


def test_allowed_changes_respects_constraints():
    X = np.array([[1, 10], [2, 20], [3, 30]])
    y = np.array([0, 1, 0])

    handler = DataHandler(
        X,
        y,
        feature_names=["f1", "f2"],
        immutable=["f1"],
        causal_inc=[("f1", "f2")],
        greater_than=[("f2", "f1")],
    )

    pre = np.array([1, 10])
    post_valid = np.array([1, 15])
    post_invalid_immutable = np.array([2, 15])

    assert handler.allowed_changes(pre, post_valid) is True
    assert handler.allowed_changes(pre, post_invalid_immutable) is False


def test_decode_empty_returns_dataframe():
    X = np.array([[1, 2], [3, 4]])
    handler = DataHandler(X, feature_names=["a", "b"])

    empty = np.empty((0, handler.encoding_width(one_hot=True)))
    dec = handler.decode(empty)
    assert isinstance(dec, pd.DataFrame)
    assert dec.empty
