# prepare_datasets.py

from dataloader import DataLoader
from helpers import *
import sys
import pandas as pd
import numpy as np
import json

def prepare_datasets(dataset_name: str, base_dir: str = '../data/') -> None:
    """
    Prepare and save train/validation/test splits plus normalization metadata
    for a given dataset (Only in case the provided csv/npy files are missing or for custom solutions).

    This will:
      1. Load one of: 'adult', 'folktables_adult', 'census', or 'law'.
      2. Filter / slice the raw DataFrame as in your original logic.
      3. Compute median absolute deviations (MAD) and per-feature min/max.
      4. One‑hot encode, normalize continuous features, and reorder columns.
      5. Split into 10% test, 10% validation, and the rest training.
      6. Save:
         - `{base_dir}/{dataset_name}-train-set.npy`
         - `{base_dir}/{dataset_name}-val-set.npy`
         - `{base_dir}/{dataset_name}-test-set.npy`
         - `{base_dir}/{dataset_name}-normalise_weights.json`
         - `{base_dir}/{dataset_name}-mad.json`

    Args:
        dataset_name (str): which dataset to prepare; one of
            'adult', 'folktables_adult', 'census', or 'law'.
        base_dir (str): directory under which to write out the .npy and .json files.
    """
    # -- LOAD & FILTER DATAFRAME ------------------------------------------------
    if dataset_name == 'adult':
        df = load_adult_income_dataset()
        cont = ['age','hours_per_week']; outcome = 'income'
        df = df[df[outcome]==0].query('age>35').append(
             df[df[outcome]==1].query('age<45'))
    elif dataset_name == 'folktables_adult':
        df = load_adult_income_dataset_folktables()
        cont = ['age','hours_per_week']; outcome = 'income'
        df = df[df[outcome]==0].query('age>35').append(
             df[df[outcome]==1].query('age<45'))
    elif dataset_name == 'census':
        df = pd.read_csv(f"{base_dir}census/census_data.csv")
        cont = ['age','wage_per_hour','capital_gains','capital_losses',
                'dividends_from_stocks','num_persons_worked_for_employer',
                'weeks_worked_in_year']; outcome = 'income'
    elif dataset_name == 'law':
        df = pd.read_csv(f"{base_dir}law/bar_pass_prediction_v2.csv")
        cont = ['decile1b','decile3','lsat','ugpa','zfygpa','zgpa']
        outcome = 'pass_bar'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # build DataLoader
    params = {'dataframe': df.copy(), 'continuous_features': cont, 'outcome_name': outcome}
    d = DataLoader(params)
    train_df = d.data_df.copy()  # after filtering above

    # -- MAD and NORMALIZATION WEIGHTS -------------------------------------------
    mad_feature_weights = d.get_mads_from_training_data(normalized=False)
    print("MAD weights:", mad_feature_weights)

    # one-hot encode & normalize
    encoded = d.one_hot_encode_data(train_df)
    arr = encoded.to_numpy()
    _, _, cat_idxs = d.get_data_params()
    cont_idxs = [i for i in range(arr.shape[1])
                 if all(i not in grp for grp in cat_idxs)]
    normalise_weights = {
        idx: [float(arr[:,idx].min()), float(arr[:,idx].max())]
        for idx in cont_idxs
    }
    encoded = d.normalize_data(encoded)

    # reorder columns so outcome is last
    cols = list(encoded.columns)
    if outcome in cols and cols[-1] != outcome:
        cols.remove(outcome)
        cols.append(outcome)
        encoded = encoded[cols]

    data = encoded.to_numpy()

    # -- SPLIT & SAVE ------------------------------------------------------------
    np.random.shuffle(data)
    n = data.shape[0]
    t = int(0.1 * n)
    test, rest = data[:t], data[t:]
    v = int(0.1 * n)
    val, train = rest[:v], rest[v:]

    np.save(f"{base_dir}{dataset_name}-train-set.npy", train)
    np.save(f"{base_dir}{dataset_name}-val-set.npy", val)
    np.save(f"{base_dir}{dataset_name}-test-set.npy", test)

    with open(f"{base_dir}{dataset_name}-normalise_weights.json", 'w') as f:
        json.dump(normalise_weights, f)

    with open(f"{base_dir}{dataset_name}-mad.json", 'w') as f:
        json.dump(mad_feature_weights, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python prepare_datasets.py <dataset_name>")
        sys.exit(1)
    prepare_datasets(sys.argv[1])
