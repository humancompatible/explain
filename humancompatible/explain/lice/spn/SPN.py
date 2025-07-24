from enum import Enum
from typing import Any

import numpy as np
from spn.algorithms.Inference import EPSILON, log_likelihood
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context, Leaf
from spn.structure.Base import Node as SPFlow_Node
from spn.structure.Base import Product, Sum, get_topological_order
from spn.structure.StatisticalTypes import MetaType

from ..data.DataHandler import DataHandler
from ..data.Features import Binary, Categorical, Contiguous, Feature, Mixed
from ..data.Types import DataLike


class NodeType(Enum):
    SUM = 0
    PRODUCT = 1
    LEAF = 2
    LEAF_CATEGORICAL = 3
    LEAF_BINARY = 4


class Node:
    """A representation of a node in an SPN"""

    def __init__(
        self,
        node: SPFlow_Node,
        feature_list: list[Feature],
        normalize: bool,
        min_density: float,
    ):
        """
        Initializes a custom Node object from an SPFlow_Node.

        This constructor wraps an SPFlow library's internal node representation
        to provide a more convenient and type-aware interface for SPN nodes.
        It extracts relevant information such as node type, scope, densities,
        and breakpoints (for continuous leaves) or weights (for sum nodes).

        Parameters:
        -----------
        node : SPFlow_Node
            The raw node object from the SPFlow library (e.g., spn.structure.Base.Leaf,
            spn.structure.Base.Product, spn.structure.Base.Sum).
        feature_list : list[Feature]
            A list of Feature objects (e.g., Contiguous, Categorical, Binary, Mixed)
            that define the characteristics of the input features. This list is
            used to determine the specific type of leaf node and its properties.
        normalize : bool
            A boolean indicating whether the data used to learn the SPN was normalized
            to a [0, 1] range. This affects how breakpoints are handled for continuous leaves.
        min_density : float
            A minimum density value to use, especially for handling edge cases
            or padding in histograms to ensure non-zero probabilities (log-likelihoods).

        Raises:
        -------
        NotImplementedError
            If a multivariate leaf node (a leaf node spanning multiple features)
            is encountered, as it's not currently supported.
        ValueError
            If an unknown or unsupported SPFlow node type is provided.
        """
        self.__normalize = normalize
        self.__min_density = min_density
        if isinstance(node, Leaf):
            self.densities = list(node.densities)
            if isinstance(node.scope, list):
                if len(node.scope) > 1:
                    raise NotImplementedError("Multivariate leaves are not supported.")
                self.scope = node.scope[0]
            else:
                self.scope = node.scope
            self.feature = feature_list[self.scope]
            if isinstance(self.feature, Categorical):
                self.type = NodeType.LEAF_CATEGORICAL
                self.options = self.feature.numeric_vals
            elif isinstance(self.feature, Binary):
                self.type = NodeType.LEAF_BINARY
            else:
                self.type = NodeType.LEAF
                # print(node.id, node.breaks, node.densities)
                self.discrete = self.feature.discrete
                if self.discrete:
                    self.breaks = [b - 0.5 for b in node.breaks]
                else:
                    self.breaks = list(node.breaks)
                dens = node.densities
                duplicate = np.isclose(dens[1:], dens[:-1], rtol=1e-10)
                self.densities = [dens[0]] + list(np.array(dens[1:])[~duplicate])
                self.breaks = (
                    [self.breaks[0]]
                    + list(np.array(self.breaks[1:-1])[~duplicate])
                    + [self.breaks[-1]]
                )
        elif isinstance(node, Product):
            self.type = NodeType.PRODUCT
        elif isinstance(node, Sum):
            self.type = NodeType.SUM
            self.weights = node.weights
        else:
            raise ValueError("")
        self.name = node.name
        self.id = node.id
        self.predecessors = node.children if hasattr(node, "children") else []

    def get_breaks_densities(
        self, span_all=True
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Returns the breakpoints and corresponding density values for a continuous
        leaf node.

        This method ensures that the breakpoints cover the entire feature range
        (if `span_all` is True) and normalizes them if the SPN was learned
        on normalized data. It's crucial for constructing piecewise linear
        functions for log-likelihood estimation.

        Parameters:
        -----------
        span_all : bool, optional
            If True, the returned breakpoints will span the entire defined range
            of the input feature (either [0, 1] if normalized, or the feature's
            original bounds). If the node's internal breaks are narrower,
            `min_density` is used to pad the outer regions. Defaults to True.

        Returns:
        --------
        tuple[np.ndarray[float], np.ndarray[float]]
            A tuple containing two NumPy arrays:
            - The first array contains the breakpoints (x-values) for the
              piecewise function, scaled to the appropriate range (0-1 if normalized,
              or original bounds if not).
            - The second array contains the corresponding density values (y-values)
              for each segment defined by the breakpoints.

        Raises:
        -------
        ValueError
            If this method is called on a node that is not a leaf node over a
            `Contiguous` feature.
        AssertionError
            If the feature bounds are not available for scaling when `span_all` is True.
        """
        if not hasattr(self, "feature") or not isinstance(self.feature, Contiguous):
            raise ValueError("Only available to leaves over contiguous features")

        density_vals = self.densities
        breaks = self.breaks

        if span_all:
            lb, ub = (0, 1) if self.__normalize else self.feature.bounds
            if lb is None or ub is None:
                raise AssertionError("SPN input variables must have fixed bounds.")
            # if histogram is narrower than the input bounds
            if lb < breaks[0]:
                breaks = [lb] + breaks
                density_vals = [self.__min_density] + density_vals
            if ub > breaks[-1]:
                breaks = breaks + [ub]
                density_vals = density_vals + [self.__min_density]

        # if the breaks are not normalized, normalize them now
        if not self.__normalize:
            breaks = self.feature.encode(breaks, normalize=True, one_hot=False)

        return np.array(breaks), np.array(density_vals)


class SPN:
    """
    A wrapper class for Sum-Product Networks (SPNs).

    This class facilitates learning an SPN from data using SPFlow, representing
    its structure in a custom `Node` format, and performing log-likelihood inference.
    It integrates with a `DataHandler` for preprocessing input data.
    """

    def __init__(
        self,
        data: DataLike,
        data_handler: DataHandler,
        normalize_data: bool = False,
        # trunk-ignore(ruff/B006)
        learn_mspn_kwargs: dict[str, Any] = {},
    ):
        """
        Initializes the SPN wrapper, learns an SPN from the provided data,
        and constructs its internal node representation.

        Parameters:
        -----------
        data : DataLike
            The input data used to learn the SPN. This can be a NumPy array
            or a pandas DataFrame, as accepted by `DataHandler`.
        data_handler : DataHandler
            An instance of a DataHandler class responsible for preprocessing
            the input data (e.g., encoding, scaling, handling feature types).
        normalize_data : bool, optional
            If True, the data will be normalized to a [0, 1] range before
            learning the SPN. This setting affects how continuous leaf node
            breakpoints are interpreted. Defaults to False.
        learn_mspn_kwargs : dict[str, Any], optional
            A dictionary of keyword arguments to be passed directly to the
            SPFlow's `learn_mspn` function. This allows customizing the
            SPN learning process (e.g., `min_instances_slice`).
            Defaults to an empty dict.
        """
        types = []
        domains = []
        self.__feature_list = data_handler.features + [data_handler.target_feature]
        for feature in self.__feature_list:
            if isinstance(feature, Contiguous):
                if feature.discrete:
                    types.append(MetaType.DISCRETE)
                    domains.append(np.arange(feature.bounds[0], feature.bounds[1] + 1))
                else:
                    types.append(MetaType.REAL)
                    domains.append(np.asarray(feature.bounds))
            elif isinstance(feature, Categorical):
                types.append(MetaType.DISCRETE)
                domains.append(np.asarray(feature.numeric_vals))
            elif isinstance(feature, Binary):
                types.append(MetaType.BINARY)
                domains.append(np.asarray([0, 1]))
            elif isinstance(feature, Mixed):
                types.append(MetaType.REAL)
                domains.append(np.asarray(feature.bounds))
            else:
                raise ValueError(f"Unsupported feature type of feature {feature}")

        context = Context(
            meta_types=types,
            domains=domains,
            feature_names=[f.name for f in self.__feature_list],
        )
        self.__normalize_data = normalize_data
        enc_data = data_handler.encode_all(
            data, normalize=normalize_data, one_hot=False
        )
        if len(domains) != data_handler.n_features + 1:
            print("recomputing domains")
            context.add_domains(enc_data)

        self.__data_handler = data_handler
        self.__mspn = learn_mspn(enc_data, context, **learn_mspn_kwargs)
        self.__nodes = [
            Node(node, self.__feature_list, self.__normalize_data, self.min_density)
            for node in get_topological_order(self.__mspn)
        ]

    def compute_ll(self, data: DataLike) -> np.ndarray[float]:
        """
        Computes the exact log-likelihood of the given data using the learned SPN.

        Parameters:
        -----------
        data : DataLike
            The input data for which to compute the log-likelihood. Can be a
            single sample (1D array) or multiple samples (2D array).

        Returns:
        --------
        np.ndarray[float]
            The log-likelihood values for the input data. Returns a scalar if
            a single sample is provided, otherwise a NumPy array of log-likelihoods.
        """
        if len(data.shape) == 1:
            return self.compute_ll(data.reshape(1, -1))[0]
        return log_likelihood(
            self.__mspn,
            self.__data_handler.encode_all(
                data, normalize=self.__normalize_data, one_hot=False
            ),
        )

    def compute_max_approx(self, data: DataLike, return_all: bool = False) -> float | dict[int, float]:
        """
        Computes an approximate log-likelihood for a single data sample
        by replacing sum operations with a max operation in the log-domain,
        just as the MIO approximation would.

        This method is useful for quickly evaluating log-likelihoods without
        the full log-sum-exp computation, which is often approximated in MIP
        contexts. It traverses the SPN in topological order.

        Parameters:
        -----------
        data : DataLike
            A single data sample (1D NumPy array or similar) for which to
            compute the approximate log-likelihood.
        return_all : bool, optional
            If True, returns a dictionary where keys are node IDs and values
            are their computed approximate log-likelihoods. If False, returns
            only the approximate log-likelihood of the root node (the final output).
            Defaults to False.

        Returns:
        --------
        float | dict[int, float]
            The approximate log-likelihood of the root node (if `return_all` is False),
            or a dictionary of approximate log-likelihoods for all nodes (if `return_all` is True).

        Raises:
        -------
        ValueError
            If more than one sample is provided in `data`.
        """
        if len(data.shape) != 1 or (data.shape[0] != 1 and len(data.shape) == 2):
            raise ValueError("Can do only one sample, so far...")

        input_data = self.__data_handler.encode_all(
            data.reshape(1, -1), normalize=self.__normalize_data, one_hot=False
        )[0]

        node_vals = {}
        for node in self.nodes:
            if node.type == NodeType.LEAF:
                for val, b in zip(
                    [self.min_density] + node.densities + [self.min_density],
                    node.breaks + [np.inf],
                ):
                    value = np.log(val)
                    if b > input_data[node.scope]:
                        break
            if node.type == NodeType.LEAF_BINARY:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            if node.type == NodeType.LEAF_CATEGORICAL:
                value = np.log(node.densities[input_data[node.scope].astype(int)])
            if node.type == NodeType.PRODUCT:
                value = sum(node_vals[n.id] for n in node.predecessors)
            if node.type == NodeType.SUM:
                value = max(
                    node_vals[n.id] + np.log(w)
                    for n, w in zip(node.predecessors, node.weights)
                )

            node_vals[node.id] = value

        if return_all:
            return node_vals
        return node_vals[self.__mspn.id]

    @property
    def nodes(self) -> list[Node]:
        """
        Returns a list of custom Node objects representing the SPN's structure,
        ordered topologically (parents appear before children).

        This property ensures that the `Node` objects are created and cached
        upon first access.

        Returns:
        --------
        list[Node]
            A list of `Node` objects, ordered such that dependencies are met
            (i.e., children nodes appear after their parents in the list).
        """
        if not hasattr(self, "SPN__nodes"):
            self.__nodes = [
                Node(node, self.__feature_list, self.__normalize_data, self.min_density)
                for node in get_topological_order(self.__mspn)
            ]
        return self.__nodes

    @property
    def min_density(self) -> float:
        """
        Returns the minimum density value used for SPN calculations.

        This value typically comes from SPFLow's `EPSILON`,
        which is a small constant to prevent log(0) issues.

        Returns:
        --------
        float
            The minimum density value (epsilon).
        """
        return EPSILON

    @property
    def out_node_id(self) -> int:
        """
        Returns the ID of the root node (output node) of the learned SPN.

        Returns:
        --------
        int
            The ID of the SPN's root node.
        """
        return self.__mspn.id

    @property
    def spn_model(self) -> SPFlow_Node:
        """
        Returns the raw SPFlow SPN model object.

        This property provides access to the underlying SPFlow representation
        of the SPN, which can be useful for direct interaction with the SPFlow
        library's functionalities.

        Returns:
        --------
        SPFlow_Node
            The root node of the learned SPFlow SPN.
        """
        return self.__mspn

    def input_scale(self, feature_i) -> float:
        """
        Returns the scaling factor for a specific input feature.

        This is relevant if the data was not normalized, in which case the
        original scale of the feature might be needed for certain computations
        (e.g., when relating changes in normalized space back to original space).

        Parameters:
        -----------
        feature_i : int
            The index of the feature for which to retrieve the input scale.

        Returns:
        --------
        float
            The input scale (1 if data was normalized, otherwise the feature's
            internal scale factor).
        """
        if self.__normalize_data:
            return 1
        else:
            return self.__data_handler.features[feature_i]._scale
