import pickle
import pandas as pd
import pytest
import os

from humancompatible.explain.lice.lice.LiCE import LiCE


@pytest.fixture
def context():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "test_context.pickle")

    with open(path, "rb") as f:
        dhandler, spn, nn, threshold = pickle.load(f)
    return dhandler, spn, nn, threshold

@pytest.fixture
def sample():
    return pd.DataFrame(
            [[0.762035, 63.000000, 0.000000, 0.096064, 3632.000000, 4.000000, 0.000000, 0.000000, 1.000000, 0.000000]],
            columns=["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
        )

def test_thresholded(context, sample):
    dhandler, spn, nn, threshold = context
    lice = LiCE(
        spn,
        nn_path=
        os.path.join(os.path.dirname(__file__), "test_nn.onnx"),
        data_handler=dhandler,
    )

    enc_sample = dhandler.encode(sample)
    prediction = nn.predict(enc_sample) > 0
    print(sample)
    print(dhandler.features)

    thresholded = lice.generate_counterfactual(
        sample.iloc[0],
        not prediction,
        ll_threshold=threshold,
        n_counterfactuals=1,
        time_limit=120,
        solver_name="appsi_highs",
    )

    assert len(thresholded[0]) == len(sample.iloc[0])
    assert (nn.predict(dhandler.encode(thresholded[0])) > 0) != prediction

def test_optimized(context, sample):
    dhandler, spn, nn, _ = context
    lice = LiCE(
        spn,
        os.path.join(os.path.dirname(__file__), "test_nn.onnx"),
        data_handler=dhandler,
    )

    enc_sample = dhandler.encode(sample)
    prediction = nn.predict(enc_sample) > 0

    optimized = lice.generate_counterfactual(
        sample.iloc[0],
        not prediction,
        ll_opt_coefficient=0.1,
        n_counterfactuals=1,
        time_limit=120,
        solver_name="appsi_highs",
    )

    assert len(optimized[0]) == len(sample.iloc[0])
    assert (nn.predict(dhandler.encode(optimized[0])) > 0) != prediction