from time import perf_counter

import numpy as np
import onnx
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io import load_onnx_neural_network
from omlt.neuralnet import FullSpaceNNFormulation
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverStatus, TerminationCondition

from ..data.DataHandler import DataHandler
from ..data.Types import DataLike
from ..spn.SPN import SPN

from .data_enc import decode_input_change, encode_input_change
from .spn_enc import encode_spn


class LiCE:
    """
    LiCE (Likely Counterfactual Explanations) class for generating
    counterfactual explanations of a neural network using a
    Sum-Product Network (SPN) to ensure plausibility.

    This class formulates and solves a Mixed-Integer Optimization (MIO) problem
    to find counterfactuals that minimize the distance to a factual instance
    while satisfying desired prediction and likelihood constraints.
    """

    MIO_EPS = 1e-6
    """
    A small epsilon value used in Mixed-Integer Optimization
    (MIO) problems, particularly for handling strict inequalities.
    """

    def __init__(self, spn: SPN, nn_path: str, data_handler: DataHandler) -> None:
        """
        Initializes the LiCE explainer.

        Parameters:
        -----------
        spn : SPN
            The Sum-Product Network (SPN) used for likelihood estimation and
            integration into the optimization problem.
        nn_path : str
            The file path to the ONNX-formatted neural network model.
        data_handler : DataHandler
            An instance of DataHandler for preprocessing and postprocessing
            of input and output data.
        """
        self.__spn = spn
        self.__nn_path = nn_path
        self.__dhandler = data_handler
        self.__t_build: float = 0.0
        self.__t_solve: float = 0.0
        self.__t_tot: float = 0.0
        self.__optimal: bool = False
        self.__loglikelihoods: list[float] = []
        self.__distances: list[float] = []
        self.__model: pyo.Model | None = None

    # TODO remove the defaults maybe?
    def __build_model(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float,
        optimize_ll: bool,
        prediction_threshold: float = 1e-4,
        ll_opt_coef: float = 0.1,
        leaf_encoding: float = "histogram",
        spn_variant: str = "lower",
    ) -> pyo.Model:
        """
        Builds the Pyomo optimization model for generating counterfactuals.

        This private method sets up the Mixed-Integer Optimization (MIO) problem
        including components for input encoding, neural network prediction,
        and (optionally) Sum-Product Network likelihood constraints/objectives.

        Parameters:
        -----------
        factual : DataLike
            The factual instance for which to generate a counterfactual.
            It's used to calculate the distance for the counterfactual.
        desired_class : bool
            The desired class for the counterfactual (True for class 1, False for class 0).
        ll_threshold : float
            The minimum log-likelihood threshold for the SPN. If a finite value
            is provided, a constraint is added to ensure the counterfactual
            has at least this log-likelihood.
        optimize_ll : bool
            If True, the log-likelihood from the SPN is optimized as part of
            the objective function.
        prediction_threshold : float, optional
            The threshold for the neural network's prediction output to determine
            the desired class. Defaults to 1e-4.
        ll_opt_coef : float, optional
            The coefficient for the log-likelihood term in the objective function
            when `optimize_ll` is True. A higher value means more emphasis on
            maximizing log-likelihood. Defaults to 0.1.
        leaf_encoding : str, optional
            The type of encoding used for leaf nodes in the SPN.
            Options include "histogram" (leading to histogram-specifc formlation)
            or values of `pw_repn` parameter of Pyomo's `Piecewise` component.
            Defaults to "histogram".
        spn_variant : str, optional
            The variant of SPN encoding to use, "lower" for a lower bound
            approximation or "upper" for approximation from above.
            Defaults to "lower".

        Returns:
        --------
        pyo.Model
            The constructed Pyomo concrete model ready for optimization.
        """
        model = pyo.ConcreteModel()

        model.input_encoding = pyo.Block()
        inputs, distance = encode_input_change(
            self.__dhandler, model.input_encoding, factual
        )

        model.predictor = OmltBlock()
        onnx_model = onnx.load(self.__nn_path)
        input_bounds = []
        input_vec = []
        for input_var in inputs:
            for var in input_var.values():
                input_vec.append(var)
                input_bounds.append(var.bounds)

        net = load_onnx_neural_network(onnx_model, input_bounds=input_bounds)
        formulation = FullSpaceNNFormulation(net)
        model.predictor.build_formulation(formulation)

        # connect the vars
        model.inputset = pyo.Set(initialize=range(len(input_vec)))

        def connect_input(mdl, i):
            return input_vec[i] == mdl.predictor.inputs[i]

        model.connect_nn_input = pyo.Constraint(model.inputset, rule=connect_input)

        sign = -1 if desired_class == 0 else 1
        model.classification = pyo.Constraint(
            expr=sign * model.predictor.outputs[0] >= prediction_threshold
        )

        spn_inputs = inputs

        if optimize_ll:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs + [int(desired_class)],
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            model.obj = pyo.Objective(
                expr=distance - ll_opt_coef * spn_outputs[self.__spn.out_node_id],
                sense=pyo.minimize,
            )
            return model

        elif ll_threshold > -np.inf:
            model.spn = pyo.Block()
            spn_outputs = encode_spn(
                self.__spn,
                model.spn,
                spn_inputs + [int(desired_class)],
                leaf_encoding=leaf_encoding,
                mio_epsilon=self.MIO_EPS,
                sum_approx=spn_variant,
            )
            model.ll_constr = pyo.Constraint(
                expr=spn_outputs[self.__spn.out_node_id] >= ll_threshold
            )

        # set up objective
        model.obj = pyo.Objective(expr=distance, sense=pyo.minimize)
        return model

    def generate_counterfactual(
        self,
        factual: DataLike,
        desired_class: bool,
        ll_threshold: float = -np.inf,
        ll_opt_coefficient: float = 0,
        n_counterfactuals: int = 1,
        solver_name: str = "gurobi",
        verbose: bool = False,
        time_limit: int = 600,
        leaf_encoding: str = "histogram",
        spn_variant: str = "lower",
        ce_relative_distance: float = np.inf,
        ce_max_distance: float = np.inf,
    ) -> tuple[bool, list[DataLike]]:
        """
        Generates one or more counterfactual explanations for a given factual instance.

        This is the main method for finding counterfactuals. It builds and solves
        the Pyomo optimization model.

        Parameters:
        -----------
        factual : DataLike
            The original instance for which to find a counterfactual.
        desired_class : bool
            The target class for the counterfactual (True for class 1, False for class 0).
        ll_threshold : float, optional
            The minimum log-likelihood for the generated counterfactuals. If set to
            a finite value, a constraint ensures the counterfactual's log-likelihood
            from the SPN meets this value. Defaults to -np.inf (no constraint).
        ll_opt_coefficient : float, optional
            If non-zero, the log-likelihood of the SPN is incorporated into the
            objective function. A positive coefficient encourages higher log-likelihood.
            Defaults to 0 (no optimization of log-likelihood).
        n_counterfactuals : int, optional
            The number of counterfactuals to generate. Currently, multiple
            counterfactuals are only supported with the 'gurobi' solver. Defaults to 1.
        solver_name : str, optional
            The name of the Pyomo-compatible solver to use (e.g., "appsi_highs", "gurobi", "cplex").
            Defaults to "gurobi".
        verbose : bool, optional
            If True, the solver's output will be printed. Defaults to False.
        time_limit : int, optional
            The maximum time (in seconds) allowed for the solver to run. Defaults to 600.
        leaf_encoding : str, optional
            The type of encoding used for leaf nodes in the SPN.
            Options include "histogram" (leading to histogram-specifc formlation)
            or values of `pw_repn` parameter of Pyomo's `Piecewise` component.
            Defaults to "histogram".
        spn_variant : str, optional
            The variant of SPN encoding to use, "lower" for a lower bound
            approximation or "upper" for approximation from above.
            Defaults to "lower".
        ce_relative_distance : float, optional
            For multiple counterfactuals (Gurobi only), this sets the relative gap
            from the optimal solution to consider other solutions in the pool.
            E.g., 0.1 means solutions within 10% of the optimal objective value.
            Defaults to np.inf (no relative distance constraint).
        ce_max_distance : float, optional
            A hard upper bound on the total cost (distance) of the counterfactual.
            Defaults to np.inf (no maximum distance constraint).

        Returns:
        --------
        tuple[bool, list[DataLike]]
            A tuple containing:
            - A boolean indicating whether an optimal solution was found (`True`)
              or if the solver terminated early/infeasibly (`False`).
            - A list of generated counterfactuals (DataLike objects). The list
              might be empty if no counterfactuals are found or if the solver
              terminates unexpectedly.

        Raises:
        -------
        NotImplementedError
            If `n_counterfactuals > 1` is requested with a solver other than 'gurobi'.
        ValueError
            If the solver terminates with an unexpected condition.
        """

        t_start = perf_counter()
        model = self.__build_model(
            factual,
            desired_class,
            ll_threshold,
            ll_opt_coefficient != 0,
            leaf_encoding=leaf_encoding,
            ll_opt_coef=ll_opt_coefficient,
            spn_variant=spn_variant,
        )
        t_built = perf_counter()
        if solver_name == "gurobi":
            opt = pyo.SolverFactory(solver_name, solver_io="python")
        else:
            opt = pyo.SolverFactory(solver_name)

        if n_counterfactuals > 1:
            if solver_name != "gurobi":
                raise NotImplementedError(
                    "Generating multiple counterfactuals is supported only for Gurobi solver"
                )
            opt.options["PoolSolutions"] = n_counterfactuals  # Store n solutions
            opt.options["PoolSearchMode"] = 2  # Systematic search for n-best solutions
            if ce_relative_distance != np.inf:
                # Accept solutions within ce_relative_distance*100% of the optimal
                opt.options["PoolGap"] = ce_relative_distance
        if ce_max_distance != np.inf:
            print("Limiting max distance by", ce_max_distance)
            model.max_dist = pyo.Constraint(
                expr=model.input_encoding.total_cost <= ce_max_distance
            )

        if "cplex" in solver_name:
            opt.options["timelimit"] = time_limit
        elif "glpk" in solver_name:
            opt.options["tmlim"] = time_limit
        elif "xpress" in solver_name:
            opt.options["soltimelimit"] = time_limit
            # Use the below instead for XPRESS versions before 9.0
            # self.solver.options['maxtime'] = TIME_LIMIT
        elif "highs" in solver_name:
            opt.options["time_limit"] = time_limit
        elif solver_name == "gurobi":
            opt.options["TimeLimit"] = time_limit
            # opt.options["Aggregate"] = 0
            # opt.options["OptimalityTol"] = 1e-3
            opt.options["IntFeasTol"] = self.MIO_EPS / 10
            opt.options["FeasibilityTol"] = self.MIO_EPS / 10
        else:
            print("Time limit not set! Not implemented for your solver")

        t_prepped = perf_counter()
        result = opt.solve(model, load_solutions=False, tee=verbose)
        t_solved = perf_counter()

        self.__t_build = t_built - t_start
        self.__t_solve = t_solved - t_prepped
        self.__model = model
        self.__loglikelihoods = []
        self.__distances = []

        if verbose:
            opt._solver_model.printStats()
            print(result)
        if result.solver.status == SolverStatus.ok:
            if result.solver.termination_condition == TerminationCondition.optimal:
                model.solutions.load_from(result)
                CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
                self.__t_tot = perf_counter() - t_start
                self.__optimal = True
                return CEs
        elif result.solver.termination_condition in [
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
            # the objective value is always bounded
        ]:
            print("Infeasible formulation")
            if verbose:
                write_iis(model, "IIS.ilp", solver="gurobi")
            self.__t_tot = perf_counter() - t_start
            self.__optimal = False
            return []
        elif (
            result.solver.status == SolverStatus.aborted
            and result.solver.termination_condition == TerminationCondition.maxTimeLimit
        ):
            print("TIME LIMIT")
            self.__optimal = False
            try:
                model.solutions.load_from(result)
            except ValueError:
                self.__t_tot = perf_counter() - t_start
                return []
            CEs = self.__get_CEs(n_counterfactuals, model, factual, opt)
            self.__t_tot = perf_counter() - t_start
            return CEs

        self.__t_tot = (perf_counter() - t_start,)
        self.__optimal = False
        # print result if it wasn't printed yet
        if not verbose:
            print(result)
        raise ValueError("Unexpected termination condition")

    def __get_CEs(
        self, n: int, model: pyo.Model, factual: np.ndarray, opt: pyo.SolverFactory
    ) -> list[DataLike]:
        """
        Retrieves the counterfactual explanations from the solved Pyomo model.

        This private helper method extracts the values of the optimized variables
        and decodes them back into the original data format, handling multiple
        solutions if available from the solver.

        Parameters:
        -----------
        n : int
            The number of counterfactuals to retrieve.
        model : pyo.Model
            The solved Pyomo concrete model.
        factual : np.ndarray
            The original factual instance (as a NumPy array) to which changes
            are applied for decoding.
        opt : pyo.SolverFactory
            The Pyomo solver factory instance, used to access solver-specific
            solution pools (e.g., Gurobi's `SolCount`).

        Returns:
        --------
        list[DataLike]
            A list of decoded counterfactual explanations. Each element in the
            list is a `DataLike` object (e.g., pandas DataFrame or NumPy array).
        """
        if n > 1:
            # this takes a lot of time for high n (~100 000)
            CEs = []
            self.__loglikelihoods = []
            self.__distances = []
            for sol in range(min(n, opt._solver_model.SolCount)):
                opt._solver_model.params.SolutionNumber = sol
                vars = opt._solver_model.getVars()
                for var in vars:
                    value = var.Xn
                    # or correct some numerical errors
                    # value = np.round(var.Xn, 10)
                    var = opt._solver_var_to_pyomo_var_map[var]
                    if var.bounds != (None, None):
                        value = np.clip(value, var.bounds[0], var.bounds[1])
                    if var.domain in [
                        pyo.Integers,
                        pyo.NonNegativeIntegers,
                        pyo.NonPositiveIntegers,
                        pyo.NegativeIntegers,
                        pyo.PositiveIntegers,
                        pyo.Binary,
                    ]:
                        # value = np.round(value)
                        value = np.round(
                            value, -np.log10(self.MIO_EPS / 10).astype(int)
                        )
                    var.value = value
                self.__distances.append(self.__model.input_encoding.total_cost.value)
                if hasattr(self.__model, "spn"):
                    self.__loglikelihoods.append(
                        self.__model.spn.node_out[self.__spn.out_node_id].value
                    )
                    # TODO move to spn enc?
                CEs.append(
                    decode_input_change(
                        self.__dhandler,
                        model.input_encoding,
                        factual,
                        # round_cont_to=int(-np.log10(self.MIO_EPS)),
                        mio_eps=self.MIO_EPS,
                        spn=self.__spn,
                        mio_spn=(
                            self.__model.spn if hasattr(self.__model, "spn") else None
                        ),
                    )
                )
            return CEs
        else:
            self.__distances.append(self.__model.input_encoding.total_cost.value)
            if hasattr(self.__model, "spn"):
                self.__loglikelihoods.append(
                    self.__model.spn.node_out[self.__spn.out_node_id].value
                )
            return [
                decode_input_change(
                    self.__dhandler,
                    model.input_encoding,
                    factual,
                    # round_cont_to=int(-np.log10(self.MIO_EPS)),
                    mio_eps=self.MIO_EPS,
                    spn=self.__spn,
                    mio_spn=self.__model.spn if hasattr(self.__model, "spn") else None,
                )
            ]

    @property
    def stats(self) -> dict[str, object]:
        """
        Returns a dictionary containing performance statistics and results
        from the last counterfactual generation attempt.

        Returns:
        --------
        dict[str, object]
            A dictionary with the following keys:
            - "time_total": Total time taken for the `generate_counterfactual` call (including CE recovery).
            - "time_solving": Time spent by the solver.
            - "time_building": Time spent building the Pyomo model.
            - "optimal": Boolean indicating if the solver found an optimal solution.
            - "ll_computed": A list of log-likelihoods for each generated counterfactual (if applicable).
            - "dist_computed": A list of distances for each generated counterfactual.
        """
        return {
            "time_total": self.__t_tot,  # with CE recovery
            "time_solving": self.__t_solve,
            "time_building": self.__t_build,
            "optimal": self.__optimal,
            "ll_computed": self.__loglikelihoods,
            "dist_computed": self.__distances,
        }

    @property
    def model(self) -> pyo.Model:
        """
        Returns the Pyomo model from the most recent `generate_counterfactual` call.

        This allows inspection of the optimization problem after it has been built
        and potentially solved.

        Returns:
        --------
        pyo.Model
            The Pyomo concrete model instance.
        """
        if self.__model is None:
            raise AttributeError("Model has not been built yet. Call generate_counterfactual first.")
        return self.__model
