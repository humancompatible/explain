from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .Features import (
    Binary,
    Categorical,
    Contiguous,
    Feature,
    Mixed,
    Monotonicity,
)
from .Types import CategValue, DataLike, FeatureID, OneDimData


class DataHandler:
    """
    Handles all data processing, transforming raw pandas DataFrames or NumPy arrays
    into a normalized and encoded format.

    This class is designed to be initialized with training data and then used to
    consistently encode all subsequent data. It supports mixed data types, where
    some values are categorical, and normalizes contiguous data to a [0, 1] range.
    The output can be either one-hot encoded or direct data with mapped categorical
    values to negative integers.
    """

    def __init__(
        self,
        X: DataLike,
        y: OneDimData | None = None,
        # trunk-ignore(ruff/B006)
        categ_map: dict[FeatureID, list[CategValue]] = {},
        # trunk-ignore(ruff/B006)
        ordered: list[FeatureID] = [],
        # trunk-ignore(ruff/B006)
        bounds_map: dict[FeatureID, tuple[int, int]] = {},
        # trunk-ignore(ruff/B006)
        discrete: list[FeatureID] = [],
        # trunk-ignore(ruff/B006)
        immutable: list[FeatureID] = [],
        # trunk-ignore(ruff/B006)
        monotonicity: dict[FeatureID, Monotonicity] = {},
        # TODO more general causality
        # trunk-ignore(ruff/B006)
        causal_inc: list[tuple[FeatureID, FeatureID]] = [],
        # trunk-ignore(ruff/B006)
        greater_than: list[tuple[FeatureID, FeatureID]] = [],
        regression: bool = False,
        feature_names: Optional[list[str]] = None,
        target_name: Optional[str] = None,
    ):
        """
        Initializes a DataHandler instance for data processing and encoding.

        Parameters:
        -----------
        X : DataLike
            Input features. Can be a pandas DataFrame or a NumPy array.
            Expected shape: (num_samples, num_features).
        y : OneDimData | None, optional
            Target feature (e.g., labels for classification or regression targets).
            Expected shape: (num_samples,). Defaults to None.
        categ_map : dict[FeatureID, list[CategValue]], optional
            A dictionary where keys are feature identifiers (indices or names) and
            values are lists of unique categorical values for that feature.
            If a list is empty, all unique values of the feature are considered
            categorical. If a list is non-empty but doesn't cover all values,
            the feature is treated as mixed. Defaults to an empty dictionary.
        ordered : list[FeatureID], optional
            A list of feature identifiers that should be treated as ordered categorical.
            Defaults to an empty list.
        bounds_map : dict[FeatureID, tuple[int, int]], optional
            A dictionary where keys are feature identifiers and values are tuples
            (min, max) defining the real bounds for contiguous features.
            Defaults to an empty dictionary.
        discrete : list[FeatureID], optional
            A list of feature identifiers that should be treated as discrete contiguous.
            Defaults to an empty list.
        immutable : list[FeatureID], optional
            A list of feature identifiers that represent immutable features (cannot be changed).
            Defaults to an empty list.
        monotonicity : dict[FeatureID, Monotonicity], optional
            A dictionary where keys are feature identifiers and values specify the
            monotonicity constraint for that feature (can only decrease or only increase).
            Defaults to an empty dictionary.
        causal_inc : list[tuple[FeatureID, FeatureID]], optional
            A list of tuples, where each tuple (cause, effect) indicates that
            an increase in 'cause' must lead to an increase in 'effect'.
            Defaults to an empty list.
        greater_than : list[tuple[FeatureID, FeatureID]], optional
            A list of tuples, where each tuple (greater, smaller) indicates that
            'greater' must be greater than 'smaller'. Defaults to an empty list.
        regression : bool, optional
            If True, the task is treated as regression; otherwise, it's classification.
            Defaults to False.
        feature_names : Optional[list[str]], optional
            A list of names for the input features. If None and `X` is a DataFrame,
            column names from `X` will be used. Defaults to None.
        target_name : Optional[str], optional
            The name of the target feature. If None and `y` is a pandas Series,
            its name will be used. If `X` is a DataFrame and `target_name` is
            provided, the target column will be extracted from `X`. Defaults to None.

        Raises:
        -------
        ValueError
            If the length of `feature_names` does not match the number of features in `X`.
        """
        if isinstance(X, pd.DataFrame):
            if target_name is not None:
                print("Taking target values from the X matrix")
                y = X[target_name]
                X = X.drop(columns=target_name)
            if feature_names is None:
                feature_names = X.columns
            X = X.to_numpy()

        if y is not None:
            if target_name is None:
                if isinstance(y, pd.Series):
                    target_name = y.name
                else:
                    target_name = "target"

            if regression:
                self.__target_feature = Contiguous(y, target_name)
            else:
                if len(np.unique(y)) > 2:
                    self.__target_feature = Categorical(y, name=target_name)
                else:
                    self.__target_feature = Binary(y, name=target_name)
                    # TODO make the target values specifiable
        else:
            self.__target_feature = None

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [None] * n_features
        if len(feature_names) != n_features:
            raise ValueError("Incorrect length of list of feature names.")

        self.__input_features: list[Feature] = []
        # stores lists of categorical values of applicable features, used for mapping to integer values
        for feat_i, feat_name in enumerate(feature_names):
            self.__input_features.append(
                self.__make_feature(
                    X[:, feat_i],
                    feat_name,
                    categ_map.get(feat_name, None),
                    bounds_map.get(feat_name, None),
                    feat_name in ordered,
                    feat_name in discrete,
                    monotone=monotonicity.get(feat_name, Monotonicity.NONE),
                    modifiable=feat_name not in immutable,
                )
            )

        self.__causal_inc = [
            (
                self.__input_features[self.feature_names.index(i)],
                self.__input_features[self.feature_names.index(j)],
            )
            for i, j in causal_inc
        ]
        self.__greater_than = [
            (
                self.__input_features[self.feature_names.index(i)],
                self.__input_features[self.feature_names.index(j)],
            )
            for i, j in greater_than
        ]

    @property
    def causal_inc(self) -> list[tuple[Feature, Feature]]:
        return self.__causal_inc

    @property
    def greater_than(self) -> list[tuple[Feature, Feature]]:
        return self.__greater_than

    def __make_feature(
        self,
        data: OneDimData,
        feat_name: Optional[str],
        categ_vals: Optional[list[CategValue]],
        real_bounds: Optional[list[CategValue]],
        ordered: bool,
        discrete: bool,
        monotone: bool,
        modifiable: bool,
    ) -> Feature:
        """
        Internal helper method to create a Feature object based on provided metadata.

        Parameters:
        -----------
        data : OneDimData
            The 1-dimensional array-like data for the feature.
        feat_name : Optional[str]
            The name of the feature.
        categ_vals : Optional[list[CategValue]]
            A list of unique categorical values for the feature. If None, the feature
            is treated as contiguous.
        real_bounds : Optional[list[CategValue]]
            A tuple (min, max) specifying the real bounds for contiguous features.
        ordered : bool
            True if the categorical feature is ordered.
        discrete : bool
            True if the contiguous feature is discrete.
        monotone : Monotonicity
            The monotonicity constraint for the feature.
        modifiable : bool
            True if the feature is modifiable.

        Returns:
        --------
        Feature
            An instance of Binary, Categorical, Contiguous, or Mixed feature.

        Raises:
        -------
        ValueError
            If an invalid feature type combination is encountered (e.g., mixed with ordered categorical).
        """
        if categ_vals is None:
            return Contiguous(
                data,
                feat_name,
                bounds=real_bounds,
                discrete=discrete,
                monotone=monotone,
                modifiable=modifiable,
            )
        else:
            if len(categ_vals) > 0:  # if predefined mapping exists
                if np.any(~np.isin(data, categ_vals)):
                    # if there are non-categorical values
                    return Mixed(
                        data,
                        categ_vals,
                        name=feat_name,
                        bounds=real_bounds,
                        monotone=monotone,
                        modifiable=modifiable,
                    )
                elif len(categ_vals) > 2:
                    return Categorical(
                        data,
                        categ_vals,
                        name=feat_name,
                        monotone=monotone,
                        modifiable=modifiable,
                        ordering=categ_vals if ordered else None,
                    )
                else:
                    return Binary(
                        data,
                        categ_vals,
                        name=feat_name,
                        monotone=monotone,
                        modifiable=modifiable,
                    )
            else:
                # fully categorical without pre-specified valuess
                if len(np.unique(data)) > 2:
                    return Categorical(
                        data, name=feat_name, monotone=monotone, modifiable=modifiable
                    )
                else:
                    return Binary(
                        data, name=feat_name, monotone=monotone, modifiable=modifiable
                    )

    @property
    def n_features(self) -> int:
        """
        The number of input features.

        Returns:
        --------
        int
            The total count of features in the input space.
        """
        return len(self.__input_features)

    @property
    def features(self) -> list[Feature]:
        """
        A list of Feature objects representing the input features.

        Returns:
        --------
        list[Feature]
            A list containing instances of Feature (e.g., Contiguous, Categorical, etc.).
        """
        return self.__input_features

    @property
    def target_feature(self) -> Feature:
        """
        The Feature object representing the target variable.

        Returns:
        --------
        Feature
            An instance of Feature (e.g., Contiguous, Categorical, or Binary)
            representing the target feature.
        """
        return self.__target_feature

    @property
    def feature_names(self) -> list[str]:
        """
        A list of names for all input features.

        Returns:
        --------
        list[str]
            A list of strings, where each string is the name of an input feature.
        """
        return [f.name for f in self.__input_features]

    def encode(
        self, X: DataLike, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        """
        Encodes the input features according to the DataHandler's configuration.

        This method transforms raw input data into a format suitable for model
        training or inference, handling normalization and one-hot encoding
        as specified.

        Parameters:
        -----------
        X : DataLike
            Input features, which can be a pandas DataFrame, pandas Series,
            or a NumPy array.
            Expected shape: (num_samples, num_features) for DataFrame/2D array,
            or (num_features,) for a single sample Series/1D array.
        normalize : bool, optional
            If True, contiguous features will be normalized to the [0, 1] range.
            Defaults to True.
        one_hot : bool, optional
            If True, categorical features will be one-hot encoded. If False,
            categorical values will be mapped to negative integers.
            Defaults to True.

        Returns:
        --------
        np.ndarray[np.float64]
            The encoded input features. The shape depends on `one_hot`:
            - If `one_hot` is True: (num_samples, total_one_hot_features)
            - If `one_hot` is False: (num_samples, num_features)

        Raises:
        -------
        ValueError
            If the input `X` has an unexpected shape or type that cannot be processed.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        if len(X.shape) == 1:
            Xmat = X.reshape(1, -1)
            return self.encode(Xmat, normalize=normalize, one_hot=one_hot)[0]

        enc = []
        for feat_i, feature in enumerate(self.__input_features):
            enc.append(
                feature.encode(X[:, feat_i], normalize, one_hot).reshape(X.shape[0], -1)
            )

        return np.concatenate(enc, axis=1).astype(np.float64)

    def encode_y(
        self, y: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        """
        Encodes the target feature (`y`) according to the DataHandler's configuration.

        This method transforms the raw target variable into a format suitable for
        model training or inference, handling normalization and one-hot encoding
        as specified.

        Parameters:
        -----------
        y : OneDimData
            The target feature data. Can be a pandas Series or a NumPy array.
            Expected shape: (num_samples,).
        normalize : bool, optional
            If True, the target feature will be normalized (if it's contiguous).
            Defaults to True.
        one_hot : bool, optional
            If True, categorical target feature will be one-hot encoded. If False,
            categorical values will be mapped to negative integers.
            Defaults to True.

        Returns:
        --------
        np.ndarray[np.float64]
            The encoded target feature. The shape depends on `one_hot` and the target type:
            - If `one_hot` is True and target is categorical: (num_samples, num_unique_target_values)
            - Otherwise: (num_samples,)
        """
        return self.__target_feature.encode(y, normalize, one_hot)

    def encode_all(self, X_all: np.ndarray, normalize: bool, one_hot: bool):
        """
        Encodes both input features and the target feature when they are
        concatenated into a single NumPy array.

        Assumes the last column of `X_all` is the target feature.

        Parameters:
        -----------
        X_all : np.ndarray
            A NumPy array where input features are in all columns except the last one,
            and the target feature is in the last column.
            Expected shape: (num_samples, num_features + 1).
        normalize : bool
            Whether to normalize contiguous features (both input and target).
        one_hot : bool
            Whether to perform one-hot encoding for categorical values (both input and target).

        Returns:
        --------
        np.ndarray[np.float64]
            The combined encoded features and target.
        """
        return np.concatenate(
            [
                self.encode(X_all[:, :-1], normalize, one_hot),
                self.encode_y(X_all[:, -1], normalize, one_hot).reshape(-1, 1),
            ],
            axis=1,
        )

    def decode(
        self,
        X: np.ndarray[np.float64],
        denormalize: bool = True,
        encoded_one_hot: bool = True,
        as_dataframe: bool = True,
    ) -> np.ndarray[np.float64]:
        """
        Decodes the encoded input features back to their original format.

        This method reverses the encoding process, denormalizing contiguous features
        and converting one-hot encoded categorical features back to their original values.

        Parameters:
        -----------
        X : np.ndarray[np.float64]
            The encoded input data matrix.
            Expected shape: (num_samples, num_encoded_features), where `num_encoded_features`
            can be higher than the original number of features due to one-hot encoding.
        denormalize : bool, optional
            If True, the denormalization process will be applied to contiguous features.
            Defaults to True.
        encoded_one_hot : bool, optional
            If True, it is assumed that the input `X` is one-hot encoded.
            Defaults to True.
        as_dataframe : bool, optional
            If True, the decoded features will be returned as a pandas DataFrame.
            If False, a NumPy array will be returned. Defaults to True.

        Returns:
        --------
        np.ndarray[np.float64] | pd.DataFrame
            The decoded features in their original format.
            - If `as_dataframe` is True: a pandas DataFrame with original feature names.
            - If `as_dataframe` is False: a NumPy array.
            Expected shape: (num_samples, num_original_features).
        """
        if X.shape[0] == 0:
            if as_dataframe:
                return pd.DataFrame([], columns=[f.name for f in self.__input_features])
            return np.empty((0, self.n_features))
        dec = []
        curr_col = 0
        for feature in self.__input_features:
            w = feature.encoding_width(encoded_one_hot)
            dec.append(
                feature.decode(X[:, curr_col : curr_col + w], denormalize, as_dataframe)
            )
            curr_col += w
        if as_dataframe:
            return pd.concat(dec, axis=1)
        return np.concatenate([x.reshape(X.shape[0], -1) for x in dec], axis=1)

    def decode_y(
        self,
        y: np.ndarray[np.float64],
        denormalize: bool = True,
        as_series: bool = True,
    ) -> np.ndarray[np.float64]:
        """
        Decodes the encoded target feature (`y`) back to its original format.

        This method reverses the encoding process for the target variable,
        denormalizing if applicable and converting one-hot encoded forms
        back to their original values.

        Parameters:
        -----------
        y : np.ndarray[np.float64]
            The encoded target feature data.
            Expected shape: (num_samples,) for non-one-hot encoded targets,
            or (num_samples, num_categorical_values) for one-hot encoded categorical targets.
        denormalize : bool, optional
            If True, denormalization will be applied to the target feature
            (if it's contiguous). Defaults to True.
        as_series : bool, optional
            If True, the decoded target feature will be returned as a pandas Series.
            If False, a NumPy array will be returned. Defaults to True.

        Returns:
        --------
        np.ndarray[np.float64] | pd.Series
            The decoded target feature data in its original format.
            - If `as_series` is True: a pandas Series with the original target name.
            - If `as_series` is False: a NumPy array.
            Expected shape: (num_samples,).
        """
        return self.__target_feature.decode(y, denormalize, as_series)

    def encoding_width(self, one_hot: bool) -> int:
        """
        Calculates the total width of the encoded input features.

        This method determines the number of columns that the encoded data
        matrix will have, considering whether one-hot encoding is applied.

        Parameters:
        -----------
        one_hot : bool
            If True, the width for one-hot encoding will be considered. If False,
            the width for direct mapping (e.g., negative integers for categories)
            will be used.

        Returns:
        --------
        int
            The total number of columns in the encoded input feature matrix.
        """
        return sum([f.encoding_width(one_hot) for f in self.__input_features])

    def allowed_changes(self, pre_vals, post_vals):
        """
        Checks if a proposed change from `pre_vals` to `post_vals` is allowed
        based on feature constraints (immutability, monotonicity) and
        defined causal/greater-than relationships.

        Parameters:
        -----------
        pre_vals : np.ndarray
            The original feature values for a single instance.
            Expected shape: (num_features,).
        post_vals : np.ndarray
            The proposed new feature values for the same instance.
            Expected shape: (num_features,).

        Returns:
        --------
        bool
            True if all changes are allowed according to the defined constraints,
            False otherwise.

        Raises:
        -------
        ValueError
            If an invalid feature type is encountered during the check of
            causal or greater-than relationships.
        """
        for f, pre, pos in zip(self.features, pre_vals, post_vals):
            if not f.allowed_change(pre, pos):
                return False

        for cause, effect in self.__causal_inc:
            cause_i = self.features.index(cause)
            pre_cause = cause.encode(pre_vals[cause_i], normalize=False, one_hot=False)
            pos_cause = cause.encode(post_vals[cause_i], normalize=False, one_hot=False)
            if isinstance(cause, Categorical):
                applied = pos_cause in cause.greater_than(pre_cause)
            elif isinstance(cause, Contiguous):
                applied = pos_cause > pre_cause
            else:
                raise ValueError("invalid feature type")
            if applied:
                effect_i = self.features.index(effect)
                pre_effect = effect.encode(
                    pre_vals[effect_i], normalize=False, one_hot=False
                )
                pos_effect = effect.encode(
                    post_vals[effect_i], normalize=False, one_hot=False
                )
                if isinstance(effect, Categorical):
                    if pos_effect not in effect.greater_than(pre_effect):
                        return False
                elif isinstance(effect, Contiguous):
                    if pos_effect <= pre_effect:
                        return False
                else:
                    raise ValueError("invalid feature type")

        for greater, smaller in self.__greater_than:
            if (
                post_vals[self.features.index(smaller)]
                > post_vals[self.features.index(greater)]
            ):
                return False
        return True
