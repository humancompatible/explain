import sys
import os
import pickle

# Go one directory up to the root (from examples/)
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from humancompatible.explain.lice.data import DataHandler, Monotonicity
from humancompatible.explain.lice.spn import SPN

# just a small wrapper of a PyTorch model, we export it to onnx later
from examples.lice.nn_model import NNModel

data = pd.read_csv(project_root+"/data/GMSC.csv", index_col=0).dropna()
X = data[data.columns[1:]]
y = data[["SeriousDlqin2yrs"]]

# set bounds on feature values
# either fixed by domain knowledge
bounds = {"RevolvingUtilizationOfUnsecuredLines": (0, 1), "DebtRatio": (0, 2)}
# or take them from the data
for col in X.columns:
    if col not in bounds:
        bounds[col] = (X[col].min(), X[col].max())

config = {
    "categ_map": {}, # categorical features, map from feature names to a list of categ values, e.g. {"sex": ["male", "female"] | if one provides an empty list with a feature name, then all unique values are taken as categories - note that training split does not have to include all values...
    "ordered": [], # list of featurenames that contain ordered categorical values, e.g. education levels
    "discrete": [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "age",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ], # contiguous discrete fearures
    "bounds_map": bounds, # bounds on all contiguous values
    "immutable": ["NumberOfDependents"], # features that cannot change
    "monotonicity": {"age": Monotonicity.INCREASING}, # features that can only increase (or decrease)
    "causal_inc": [], # causal increase, pairs of features where if one increases, the other one must increase as well, e.g. [("education", "age")]
    "greater_than": [], # inter-feature dependence, one feature must always be > anohter feature, a list of pairs, e.g. [("# total missed payments",  "# missed payments last year")]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dhandler = DataHandler(
    X_train,
    y_train,
    **config,
    # optionally, one can provide the list of feature names (and target name) but here we pass pandas dataframe (and a series) with named columns, which will be taken as feature names
)

# finally, encode the input data
X_train_enc = dhandler.encode(X_train, normalize=True, one_hot=True)
y_train_enc = dhandler.encode_y(y_train, one_hot=False)

nn = NNModel(dhandler.encoding_width(True), [10], 1)
nn.train(X_train_enc, y_train_enc, epochs=50)
# output it to ONNX file
nn.save_onnx("test_nn.onnx")

# this can take long...
# setting "min_instances_slice":1000 argument leads to faster training by allowing leaves to be formed on >=1000 samples (default is 200)

# data should be numpy array of original (non-encoded) values, should include the target as last feature
spn_data = np.concatenate([X_train.values, y_train.values], axis=1)
spn = SPN(spn_data, dhandler, normalize_data=False, learn_mspn_kwargs={"min_instances_slice":5000})

lls = spn.compute_ll(spn_data)
quartile_ll = np.quantile(lls, 0.25) # threhsold on CE likelihood

with open("test_context.pickle", "wb") as f:
    pickle.dump((dhandler, spn, nn, quartile_ll), f)

np.random.seed(1)
i = np.random.randint(X_test.shape[0])

sample = X_test.iloc[i]
print(sample)
