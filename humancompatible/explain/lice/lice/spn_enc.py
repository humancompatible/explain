import numpy as np
import pyomo.environ as pyo

from ..spn.SPN import SPN, NodeType

# from scipy.special import logsumexp


# issues with binding variables in lambda functions for constraints
# trunk-ignore-all(ruff/B023)


def encode_histogram_as_pwl(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    encoding_type: str = "LOG",
) -> pyo.Piecewise:
    breakpoints = [breaks[0]]
    for b in breaks[1:-1]:
        breakpoints += [b, b]
    breakpoints.append(breaks[-1])

    doubled_vals = []
    for d in vals:
        doubled_vals += [d, d]

    return pyo.Piecewise(
        out_var,
        in_var,
        pw_pts=breakpoints,
        pw_constr_type="EQ",
        pw_repn=encoding_type,
        f_rule=list(doubled_vals),
    )


def encode_histogram(
    breaks: list[float],
    vals: list[float],
    in_var: pyo.Var,
    out_var: pyo.Var,
    mio_block: pyo.Block,
    mio_epsilon: float,
):
    n_bins = len(vals)
    M = max(1, breaks[-1] - breaks[0])

    mio_block.bins = pyo.Set(initialize=list(range(n_bins)))
    mio_block.not_in_bin = pyo.Var(mio_block.bins, domain=pyo.Binary)
    mio_block.one_bin = pyo.Constraint(
        expr=sum(mio_block.not_in_bin[i] for i in mio_block.bins) == n_bins - 1
    )

    mio_block.lower = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M >= breaks[bin_i] - in_var,
    )
    mio_block.upper = pyo.Constraint(
        mio_block.bins,
        rule=lambda b, bin_i: b.not_in_bin[bin_i] * M
        >= in_var - breaks[bin_i + 1] + mio_epsilon,
    )

    mio_block.output = pyo.Constraint(
        expr=sum((1 - mio_block.not_in_bin[i]) * vals[i] for i in range(n_bins))
        == out_var
    )


def encode_spn(
    spn: SPN,
    mio_spn: pyo.Block,
    input_vars: list[list[pyo.Var] | pyo.Var],
    leaf_encoding: str = "histogram",
    mio_epsilon: float = 1e-6,
    sum_approx: str = "lower",
) -> pyo.Var:

    """
    Encodes a Sum-Product Network (SPN) into a Mixed-Integer Programming (MIP)
    formulation within a Pyomo model.

    This function constructs the necessary Pyomo variables and constraints to
    represent the log-likelihood computation of the SPN. It supports different
    types of SPN nodes (leaf, product, sum) and various encoding strategies
    for leaf nodes and approximations for sum nodes in the log-domain.

    Parameters:
    -----------
    spn : SPN
        The Sum-Product Network (SPN) object to be encoded into the MIP model.
        This object contains the structure (nodes, edges, weights) and parameters
        (densities, breakpoints) of the SPN.
    mio_spn : pyo.Block
        A Pyomo Block object where the MIP formulation of the SPN will be built.
        All Pyomo components (variables, constraints) related to the SPN
        encoding will be added to this block.
    input_vars : list[list[pyo.Var] | pyo.Var]
        A list of Pyomo variables representing the inputs to the SPN. Each
        element in the list corresponds to an input feature as defined by the
        SPN's scope. For continuous features, it's typically a single `pyo.Var`.
        For categorical features, it's expected to be a list of binary `pyo.Var`s
        representing a one-hot encoding of the categories. The ordering of
        these variables should match the input feature ordering of the SPN.
    leaf_encoding : str, optional
        Specifies the method for encoding continuous leaf nodes:
        - "histogram": Models the histogram distributions directly using our
          definitions.
        - Other values (e.g., "LOG") refer to more general piecewise linear
          function approximations, leveraging Pyomo's `Piecewise` component
          for log-likelihood estimation.
        Defaults to "histogram".
    mio_epsilon : float, optional
        A small positive value used for numerical stability in MIP constraints,
        particularly for sharp inequalities or when dealing with floating-point
        comparisons. It helps to prevent issues with strict inequalities by
        introducing a small tolerance. Defaults to 1e-6.
    sum_approx : str, optional
        Specifies the approximation method for sum nodes in the log-domain.
        Sum nodes compute a weighted sum of probabilities, which translates
        to a log-sum-exp operation, a non-linear and non-convex function.
        - "lower": Implements a lower-bound approximation for the log-sum-exp
          using Big-M constraints and binary indicator variables. This
          effectively models the max operation over the weighted log-likelihoods
          of the children, providing a lower bound on the true log-sum-exp.
        - "upper": Implements an upper-bound approximation for the log-sum-exp.
          This also uses Big-M constraints and binary variables, providing
          an upper bound on the true log-sum-exp (e.g., by considering the
          maximum weighted log-likelihood plus a constant based on the number of children).
        Defaults to "lower".

    Returns:
    --------
    pyo.Var
        An indexed Pyomo variable (`mio_spn.node_out`) containing the computed
        log-likelihood values for each node in the SPN. The variable is
        indexed by the `node.id` of the SPN nodes.

    Raises:
    -------
    ValueError
        - If `input_vars` for a categorical leaf node is not a list of binary
          variables (i.e., not one-hot encoded).
        - If `sum_approx` is not one of "lower" or "upper".

    Notes:
    ------
    - The encoding of sum nodes as an approximation (lower/upper bound) is a
      technique in MIP formulations of SPNs to handle the non-linear
      log-sum-exp operation, making the problem tractable for MILP solvers.

    """
    node_ids = [node.id for node in spn.nodes]

    # node_type_ids = {t: [] for t in NodeType}
    # for node in spn.nodes:
    #     node_type_ids[node.type].append(node.id)
    #     node_ids.append(node.id)

    # mio_spn.node_type_sets = {
    #     t: pyo.Set(initialize=ids) for t, ids in node_type_ids.items()
    # }
    mio_spn.node_set = pyo.Set(initialize=node_ids)

    # values are log likelihoods - almost always negative - except in narrow peaks that go above 1
    # mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.NonPositiveReals)
    mio_spn.node_out = pyo.Var(mio_spn.node_set, within=pyo.Reals)
    # print(mio_spn.node_set, node_ids)

    for node in spn.nodes:
        if node.type == NodeType.LEAF:
            # in_var = mio_spn.input[node.scope]
            in_var = input_vars[node.scope]


            breakpoints, densities = node.get_breaks_densities(span_all=True)
            log_densities = np.log(densities)

            if leaf_encoding == "histogram":
                hist_block = pyo.Block()
                mio_spn.add_component(f"HistLeaf{node.id}", hist_block)
                encode_histogram(
                    breakpoints,
                    log_densities,
                    in_var,
                    mio_spn.node_out[node.id],
                    hist_block,
                    mio_epsilon,  # * spn.input_scale(node.scope),
                )
            else:
                pw_constr = encode_histogram_as_pwl(
                    breakpoints,
                    log_densities,
                    in_var,
                    mio_spn.node_out[node.id],
                    leaf_encoding,
                )
                mio_spn.add_component(f"PWLeaf{node.id}", pw_constr)

        elif node.type == NodeType.LEAF_CATEGORICAL:
            dens_ll = np.log(node.densities)
            in_vars = input_vars[node.scope]

            if isinstance(in_vars, pyo.Var):
                in_vars = [in_vars[k] for k in sorted(in_vars.keys())]

            if len(in_vars) <= 1:  # TODO make this more direct, not fixed to 1
                raise ValueError(
                    "The categorical values should be passed as a list of binary variables, representing a one-hot encoding."
                )
            # Do checks that the vars are binary?
            # check if the histogram always contains all values?
            # TODO use expr parameter of Constraint maker, instead of the rule=lambdas?

            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == sum(var * dens for var, dens in zip(in_vars, dens_ll))
                )
            )
            mio_spn.add_component(f"CategLeaf{node.id}", constr)
        elif node.type == NodeType.LEAF_BINARY:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == (1 - input_vars[node.scope]) * np.log(node.densities[0])
                    + input_vars[node.scope] * np.log(node.densities[1])
                )
            )
            mio_spn.add_component(f"BinLeaf{node.id}", constr)
        elif node.type == NodeType.PRODUCT:
            constr = pyo.Constraint(
                rule=lambda b: (
                    b.node_out[node.id]
                    == sum(b.node_out[ch.id] for ch in node.predecessors)
                )
            )
            mio_spn.add_component(f"ProdConstr{node.id}", constr)
        elif node.type == NodeType.SUM:
            # Sum node - approximated in log domain by max
            preds_set = [ch.id for ch in node.predecessors]
            n_preds = len(node.predecessors)
            weights = {ch.id: w for ch, w in zip(node.predecessors, node.weights)}

            # TODO testing this, if it works well, fit it in correctly
            M_sum = 100  # hope this is enough
            slack_inds = pyo.Var(preds_set, domain=pyo.Binary)
            mio_spn.add_component(f"SumSlackIndicators{node.id}", slack_inds)
            if sum_approx == "lower":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id]
                        + np.log(weights[pre_id])
                        + M_sum * slack_inds[pre_id]
                    ),
                )
            elif sum_approx == "upper":
                slacking = pyo.Constraint(
                    preds_set,
                    rule=lambda b, pre_id: (
                        b.node_out[node.id]
                        <= b.node_out[pre_id]
                        + (  # approximate by the bound on logsumexp
                            np.log(weights[pre_id] * n_preds)
                            if weights[pre_id] * n_preds < 1
                            else 0  # or by using the fact it is a mixture
                        )
                        + M_sum * slack_inds[pre_id]
                    ),
                )
            else:
                raise ValueError('sum_approx must be one of ["upper", "lower"]')
            mio_spn.add_component(f"SumSlackConstr{node.id}", slacking)
            one_tight = pyo.Constraint(
                expr=sum(slack_inds[i] for i in preds_set) == n_preds - 1
            )
            mio_spn.add_component(f"SumTightConstr{node.id}", one_tight)

    return mio_spn.node_out
