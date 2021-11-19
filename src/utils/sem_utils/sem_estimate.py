from collections import OrderedDict
from typing import Callable, Dict
from networkx import MultiDiGraph

from numpy.random import randn

from src.utils.dag_utils.graph_functions import query_common_cause

from ..utilities import select_sample


def auto_sem_hat(
    summary_graph_node_parents: Dict[str, tuple],
    independent_causes: Dict[str, bool],
    emission_functions: dict,
    transition_functions: dict = None,
) -> classmethod:
    """
    This function is used to automatically create the estimates for the edges in a given graph.

    Parameters
    ----------
    summary_graph_node_parents: Iterable
        A dictionary with causally ordered keys for the summary graph, with values containing the parents of the variables in that graph
    emission_functions : dict
        A dictionary of fitted emission functions.
    transition_functions : dict
        A dictionary of fitted transition functions.
    independent_causes: bool
        Tells the function if the first varible should be treated as an independent cause node (instrument variable) and thus has no incoming edges from the previous time-slice. This is always true for t=0. However, independent causes can appear at any time-slice including on the target node itself.

    Notes
    -----
    1. We have _NOT_ covered all network topologies with this function. Beware.
    2. This function will eventually be passed the full adjacency matrix for the graph and we will essence create a simulator for the entries in that matrix.

    Returns
    -------
    classmethod
        A SEM estimate found using observational data only; used in finding the optimal intervention set and values for CBO and DCBO.
    """

    class SEMHat:
        def __init__(self) -> None:
            self.common_cause_children = query_common_cause(summary_graph_node_parents)

        @staticmethod
        def _make_white_noise_fnc() -> Callable:
            #  Samples the exogenous model which we take to be white noise
            return lambda: randn()

        @staticmethod
        def _make_static_fnc(moment: int, common_cause_child: str = None) -> Callable:
            #  No temporal dependence
            return lambda t, emit_input_var, sample: (
                emission_functions[t][emit_input_var][common_cause_child]
                if common_cause_child
                else emission_functions[t][emit_input_var]
            ).predict(select_sample(sample, emit_input_var, t))[moment]

        @staticmethod
        def _make_dynamic_fnc(moment: int, common_cause_child: str = None) -> Callable:
            #  Temporal dependence
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_functions[
                    transfer_input_vars
                ].predict(select_sample(sample, transfer_input_vars, t - 1))[moment]
                + (
                    emission_functions[t][emit_input_vars][common_cause_child]
                    if common_cause_child
                    else emission_functions[t][emit_input_vars]
                ).predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        @staticmethod
        def _make_only_dynamic_transfer_fnc(moment: int) -> Callable:
            return lambda t, transfer_input_vars, _, sample: transition_functions[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        def static(self, moment: int):
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Assumes that the keys are causally ordered
            for v in summary_graph_node_parents:
                if not summary_graph_node_parents[v] or independent_causes[v]:
                    # This is how CBO 'views' the graph.
                    # Always sample from the exogenous model at the root node and independent cause variables -- unless other models are specified
                    # TODO: replace with marginals
                    f[v] = self._make_white_noise_fnc()
                else:
                    # Check if v has a common cause with other vertices in this time-slice sub-graph
                    if v in self.common_cause_children:
                        f[v] = self._make_static_fnc(moment, common_cause_child=v)
                    else:
                        assert len(self.common_cause_children) == 0
                        f[v] = self._make_static_fnc(moment)
            return f

        def dynamic(self, moment: int):
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Variables are causally ordered
            for i, v in enumerate(summary_graph_node_parents):
                if i == 0 and independent_causes[v]:
                    """
                    Variable at the root of the time-slice, without any time dependence
                  t-1   t
                    o   x Node at time t (assumes only _ONE_ independent cause)
                        |
                        v
                        o Child node at time t
                    """
                    # TODO: replace with marginals
                    f[v] = self._make_white_noise_fnc()
                elif i == 0 and not independent_causes[v]:
                    """
                    Root node in the time-slice, with time dependence
                  t-1   t
                    o-->x Node at time t with dependence from time t-1
                        |
                        v
                        o
                    """
                    assert not summary_graph_node_parents[v], summary_graph_node_parents
                    f[v] = self._make_only_dynamic_transfer_fnc(moment)
                elif i > 0 and independent_causes[v] and not summary_graph_node_parents[v]:
                    """
                    Variable in the time-slice, without any time dependence
                        o Node at time t
                    x   |
                      \ v
                        o Child node at time t
                    """
                    # TODO: replace with marginals
                    f[v] = self._make_white_noise_fnc()
                elif i > 0 and not summary_graph_node_parents[v]:
                    """
                    Node in the time-slice, with time dependence
                  t-1   t
                        o
                        ^
                        |
                    o-->x Node at time t with dependence from time t-1
                    """
                    f[v] = self._make_only_dynamic_transfer_fnc(moment)
                else:
                    # Check if v has a common cause with other vertices in this time-slice sub-graph
                    if v in self.common_cause_children:
                        f[v] = self._make_dynamic_fnc(moment, common_cause_child=v)
                    else:
                        assert len(self.common_cause_children) == 0
                        f[v] = self._make_dynamic_fnc(moment)
            return f

    return SEMHat


def build_sem_hat(G: MultiDiGraph, emission_fncs: dict, transition_fncs: dict = None,) -> classmethod:
    """
    This function is used to automatically create the SEM -function estimates for the edges in a given graph.

    Parameters
    ----------
    G : MultiDiGraph
        Causal graphical model
    emission_functions : dict
        A dictionary of fitted emission functions (roughly most horizontal edges in the DAG).
    transition_functions : dict
        A dictionary of fitted transition functions (roughly most vertical edges in the DAG)..

    Notes
    -----
    We have _NOT_ covered all network topologies with this function. Beware.

    Returns
    -------
    classmethod
        A SEM estimate found using observational data only; used in finding the optimal intervention set and values for CBO and DCBO.
    """

    class SEMHat:
        def __init__(self):
            self.G = G
            nodes = G.nodes()
            n_t = len(nodes) / G.T  # Number of nodes per time-slice
            assert n_t.is_integer()
            self.n_t = int(n_t)

        @staticmethod
        def _make_marginal() -> Callable:
            #  Assigns the KDE for the marginal
            return lambda t, margin_id: emission_fncs[t][margin_id].sample()

        @staticmethod
        def _make_emit_fnc(moment: int) -> Callable:
            #  Within time-slice emission only
            return lambda t, emit_input_vars, sample: emission_fncs[t][emit_input_vars].predict(
                select_sample(sample, emit_input_vars, t)
            )[moment]

        @staticmethod
        def _make_trans_fnc(moment: int) -> Callable:
            #  Only transition between time-slices (only valid for first-order Markov assumption)
            return lambda t, transfer_input_vars, _, sample: transition_fncs[transfer_input_vars].predict(
                select_sample(sample, transfer_input_vars, t - 1)
            )[moment]

        @staticmethod
        def _make_emit_plus_trans_fnc(moment: int) -> Callable:
            #  Transition plus within-slice emission(s)
            return (
                lambda t, transfer_input_vars, emit_input_vars, sample: transition_fncs[transfer_input_vars].predict(
                    select_sample(sample, transfer_input_vars, t - 1)
                )[moment]
                + emission_fncs[t][emit_input_vars].predict(select_sample(sample, emit_input_vars, t))[moment]
            )

        def static(self, moment: int) -> OrderedDict:
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Assumes that the keys are causally ordered
            for v in list(self.G.nodes)[: self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    f[vv] = self._make_marginal()
                else:
                    f[vv] = self._make_emit_fnc(moment)
            return f

        def dynamic(self, moment: int) -> OrderedDict:
            assert moment in [0, 1], moment
            # SEM functions
            f = OrderedDict()
            # Variables are causally ordered
            for v in list(self.G.nodes)[self.n_t : 2 * self.n_t]:
                vv = v.split("_")[0]
                if self.G.in_degree[v] == 0:
                    # Single source node
                    f[vv] = self._make_marginal()
                elif all(int(vv.split("_")[1]) + 1 == int(v.split("_")[1]) for vv in G.predecessors(v)):
                    # Depends only on incoming transition edge(s)
                    f[vv] = self._make_trans_fnc(moment)
                elif all(vv.split("_")[1] == v.split("_")[1] for vv in G.predecessors(v)):
                    # Depends only on incoming emission edge(s) from this time-slice
                    f[vv] = self._make_emit_fnc(moment)
                else:
                    # Depends incoming emission AND transition edges
                    f[vv] = self._make_emit_plus_trans_fnc(moment)
            return f

    return SEMHat
