from copy import deepcopy
from random import choice
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from networkx.classes.multidigraph import MultiDiGraph
from numpy.core.multiarray import ndarray
from numpy.core.numeric import nan
from src.bayes_opt.cost_functions import define_costs
from src.utils.dag_utils.graph_functions import get_independent_causes, get_summary_graph_node_parents
from src.utils.sem_utils.emissions import fit_sem_emit_fncs, get_emissions_input_output_pairs
from src.utils.sequential_intervention_functions import (
    evaluate_target_function,
    get_interventional_grids,
    make_sequential_intervention_dictionary,
)
from src.utils.utilities import (
    check_reshape_add_data,
    create_intervention_exploration_domain,
    initialise_DCBO_parameters_and_objects_filtering,
    initialise_global_outcome_dict_new,
    initialise_optimal_intervention_level_list,
    make_column_shape_2D,
)


class RooTest:
    """
    Base class with common operations, variables and functions for all BO methods.
    """

    def __init__(
        self,
        G: str,
        sem: classmethod,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict = None,
        exploration_sets: list = None,
        estimate_sem: bool = False,
        base_target_variable: str = "Y",
        task: str = "min",
        cost_type: int = 1,
        use_mc: bool = False,
        number_of_trials=10,
        ground_truth: list = None,
        n_restart: int = 1,
        debug_mode: bool = False,
        online: bool = False,
        num_anchor_points: int = 100,
        args_sem=None,
        manipulative_variables=None,
        change_points: list = None,
    ):
        if args_sem is None and change_points is None:
            true_sem = sem()
        elif args_sem and change_points is None:
            true_sem = sem(args_sem[0], args_sem[1])
        else:
            true_sem = sem(change_points.index(True))

        # These will be used in the target function evaluation
        self.true_initial_sem = true_sem.static()  # for t = 0
        self.true_sem = true_sem.dynamic()  # for t > 0
        self.make_sem_hat = make_sem_estimator

        assert isinstance(G, MultiDiGraph)
        self.G = G
        self.debug_mode = debug_mode
        # Number of optimization restart for GPs
        self.n_restart = n_restart
        self.online = online
        self.use_mc = use_mc

        # Total time-steps and sample count per time-step
        _, self.T = observation_samples[list(observation_samples.keys())[0]].shape
        self.observational_samples = observation_samples
        self.base_target_variable = base_target_variable  # This has to be reflected in the CGM
        self.index_name = 0
        self.number_of_trials = number_of_trials

        #  Induced sub-graph on the nodes in the first time-slice -- it doesn't matter which time-slice we consider since one of the main assumptions is that time-slice topology does not change in the DBN.
        time_slice_vars = observation_samples.keys()
        self.summary_graph_node_parents, self.causal_order = get_summary_graph_node_parents(time_slice_vars, G)
        #  Checks what vars in DAG (if any) are independent causes
        self.independent_causes = get_independent_causes(time_slice_vars, G)

        self.node_children, self.node_parents, self.emission_pairs = get_emissions_input_output_pairs(self.T, self.G)

        self.sem_emit_fncs = fit_sem_emit_fncs(self.observational_samples, self.emission_pairs)

        # Check that we are either minimising or maximising the objective function
        assert task in ["min", "max"], task
        self.task = task
        if task == "min":
            self.blank_val = 1e7  # Positive infinity
        elif task == "max":
            self.blank_val = -1e7  # Negative infinity

        # Instantiate blanket that will form final solution
        self.optimal_blanket = make_sequential_intervention_dictionary(self.G, self.T)

        # Contains all values a assigned as the DCBO walks through the graph;
        # optimal intervention level are assigned at the same temporal level,
        # for which we then use spatial SEMs to predict the other variable levels on that time-slice.
        self.assigned_blanket = deepcopy(self.optimal_blanket)
        self.empty_intervention_blanket = make_sequential_intervention_dictionary(self.G, self.T)

        # Canonical manipulative variables
        if manipulative_variables is None:
            self.manipulative_variables = list(
                filter(lambda k: self.base_target_variable not in k, self.observational_samples.keys(),)
            )
        else:
            self.manipulative_variables = manipulative_variables

        self.interventional_variable_limits = intervention_domain
        assert self.manipulative_variables == list(intervention_domain.keys())
        assert isinstance(exploration_sets, list)
        self.exploration_sets = exploration_sets

        # Extract all target variables from the causal graphical model
        self.all_target_variables = list(filter(lambda k: self.base_target_variable in k, self.G.nodes))

        # Get the interventional grids
        self.interventional_grids = get_interventional_grids(
            self.exploration_sets, intervention_domain, size_intervention_grid=100
        )

        # Objective function params
        self.bo_model = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}

        # Store true objective function
        self.ground_truth = ground_truth

        # Number of points where to evaluate acquisition function
        self.num_anchor_points = num_anchor_points

        # Assigned during optimisation
        self.mean_function = deepcopy(self.bo_model)
        self.variance_function = deepcopy(self.bo_model)

        # Store the dict for mean and var values computed in the acquisition function
        self.mean_dict_store = {t: {es: {} for es in self.exploration_sets} for t in range(self.T)}
        self.var_dict_store = deepcopy(self.mean_dict_store)

        # For logging
        self.sequence_of_interventions_during_trials = [[] for _ in range(self.T)]

        # Initial optimal solutions
        if intervention_samples:
            # Provide initial interventional data
            (
                initial_optimal_sequential_intervention_sets,
                initial_optimal_target_values,
                initial_optimal_sequential_intervention_levels,
                self.interventional_data_x,
                self.interventional_data_y,
            ) = initialise_DCBO_parameters_and_objects_filtering(
                self.exploration_sets,
                intervention_samples,
                self.base_target_variable,
                self.T,
                self.task,
                index_name=0,
                nr_interventions=None,  # There are interventions, we just don't sub-sample them.
            )
        else:
            # No initial interventional data
            initial_optimal_sequential_intervention_sets = [choice(self.exploration_sets)] + (self.T - 1) * [None]
            initial_optimal_target_values = self.T * [None]
            initial_optimal_sequential_intervention_levels = self.T * [None]
            self.interventional_data_x = deepcopy(self.bo_model)
            self.interventional_data_y = deepcopy(self.bo_model)

        assert (
            len(initial_optimal_sequential_intervention_levels)
            == len(initial_optimal_target_values)
            == len(initial_optimal_sequential_intervention_levels)
            == self.T
        )

        # Dict indexed by the global exploration sets, stores the best
        self.outcome_values = initialise_global_outcome_dict_new(self.T, initial_optimal_target_values, self.blank_val)
        self.optimal_outcome_values_during_trials = [[] for _ in range(self.T)]

        self.optimal_intervention_levels = initialise_optimal_intervention_level_list(
            self.T,
            self.exploration_sets,
            initial_optimal_sequential_intervention_sets,
            initial_optimal_sequential_intervention_levels,
            number_of_trials,
        )
        self.best_initial_es = initial_optimal_sequential_intervention_sets[0]  # 0 indexes the first time-step

        # Target functions for Bayesian optimisation - ground truth
        self.target_functions = {t: {es: None for es in self.exploration_sets} for t in range(self.T)}

        for temporal_index in range(self.T):
            for es in self.exploration_sets:
                self.target_functions[temporal_index][es] = evaluate_target_function(
                    self.true_initial_sem, self.true_sem, self.G, es, self.observational_samples.keys(), self.T,
                )

        # Parameter space for optimisation
        self.intervention_exploration_domain = create_intervention_exploration_domain(
            self.exploration_sets, intervention_domain,
        )

        # Optimisation specific parameters to initialise
        self.trial_type = [[] for _ in range(self.T)]  # If we observed or intervened during the trial
        self.cost_functions = define_costs(self.manipulative_variables, self.base_target_variable, cost_type)
        self.per_trial_cost = [[] for _ in range(self.T)]
        self.optimal_intervention_sets = [None for _ in range(self.T)]

        # Acquisition function specifics
        self.y_acquired = {es: None for es in self.exploration_sets}
        self.corresponding_x = deepcopy(self.y_acquired)
        self.estimate_sem = estimate_sem
        if self.estimate_sem:
            self.assigned_blanket_hat = deepcopy(self.optimal_blanket)

    def _plot_surrogate_model(
        self, temporal_index,
    ):
        # Plot model
        for es in self.exploration_sets:
            if len(es) == 1:
                inputs = np.asarray(self.interventional_grids[es])
                if self.bo_model[temporal_index][es] is not None:
                    mean, var = self.bo_model[temporal_index][es].predict(self.interventional_grids[es])
                    print("\n\t\t[1] The BO model exists for ES: {} at t == {}.\n".format(es, temporal_index))
                    print("Assigned blanket", self.assigned_blanket)
                else:
                    if temporal_index > 0 and isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                        mean = np.zeros_like(self.interventional_grids[es])
                        var = np.ones_like(self.interventional_grids[es])
                    else:
                        mean = self.mean_function[temporal_index][es](self.interventional_grids[es])
                        var = self.variance_function[temporal_index][es](self.interventional_grids[es]) + np.ones_like(
                            self.variance_function[temporal_index][es](self.interventional_grids[es])
                        )

                true = make_column_shape_2D(self.ground_truth[temporal_index][es])

                if (
                    self.interventional_data_x[temporal_index][es] is not None
                    and self.interventional_data_y[temporal_index][es] is not None
                ):
                    plt.scatter(
                        self.interventional_data_x[temporal_index][es], self.interventional_data_y[temporal_index][es],
                    )

                plt.fill_between(inputs[:, 0], (mean - var)[:, 0], (mean + var)[:, 0], alpha=0.2)
                plt.plot(
                    inputs, mean, "b", label="$do{}$ at $t={}$".format(es, temporal_index),
                )
                plt.plot(inputs, true, "r", label="True at $t={}$".format(temporal_index))
                plt.legend()
                plt.show()

    def _update_opt_params(self, it: int, temporal_index: int, best_es: tuple) -> None:

        # When observed append previous optimal values for logs
        # Outcome values at previous step

        self.outcome_values[temporal_index].append(self.outcome_values[temporal_index][-1])

        if it == 0:
            # Special case for first time index
            # At first trial assign an outcome values that is the same as the initial value
            self.optimal_outcome_values_during_trials[temporal_index].append(self.outcome_values[temporal_index][-1])

            if self.interventional_data_x[temporal_index][best_es] is None:
                self.optimal_intervention_levels[temporal_index][best_es][it] = nan

            self.per_trial_cost[temporal_index].append(0.0)

        elif it > 0:
            # Get previous one cause we are observing thus we no need to recompute it
            self.optimal_outcome_values_during_trials[temporal_index].append(
                self.optimal_outcome_values_during_trials[temporal_index][-1]
            )
            self.optimal_intervention_levels[temporal_index][best_es][it] = self.optimal_intervention_levels[
                temporal_index
            ][best_es][it - 1]
            # The cost of observation is the same as the previous trial.
            self.per_trial_cost[temporal_index].append(self.per_trial_cost[temporal_index][-1])

    def _check_new_point(self, best_es, temporal_index):
        assert best_es is not None, (best_es, self.y_acquired)
        assert best_es in self.exploration_sets

        # Check that new intervention point is in the allowed intervention domain
        assert self.intervention_exploration_domain[best_es].check_points_in_domain(self.corresponding_x[best_es])[0], (
            best_es,
            temporal_index,
            self.y_acquired,
            self.corresponding_x,
        )

    def _check_optimization_results(self, temporal_index):
        # Check everything went well with the trials
        assert len(self.optimal_outcome_values_during_trials[temporal_index]) == self.number_of_trials, (
            len(self.optimal_outcome_values_during_trials[temporal_index]),
            self.number_of_trials,
        )
        assert len(self.per_trial_cost[temporal_index]) == self.number_of_trials, len(self.per_trial_cost)

        if temporal_index > 0:
            assert all(
                len(self.optimal_intervention_levels[temporal_index][es]) == self.number_of_trials
                for es in self.exploration_sets
            ), [len(self.optimal_intervention_levels[temporal_index][es]) for es in self.exploration_sets]

        assert self.optimal_intervention_sets[temporal_index] is not None, (
            self.optimal_intervention_sets,
            self.optimal_intervention_levels,
            temporal_index,
        )


    def _safe_optimization(self, temporal_index, exploration_set, bound_var=1e-02, bound_len=20.0):
        if self.bo_model[temporal_index][exploration_set].model.kern.variance[0] < bound_var:
            self.bo_model[temporal_index][exploration_set].model.kern.variance[0] = 1.0

        if self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] > bound_len:
            self.bo_model[temporal_index][exploration_set].model.kern.lengthscale[0] = 1.0

    def _get_updated_interventional_data(self, new_interventional_data_x, y_new, best_es, temporal_index):
        data_x, data_y = check_reshape_add_data(
            self.interventional_data_x,
            self.interventional_data_y,
            new_interventional_data_x,
            y_new,
            best_es,
            temporal_index,
        )
        self.interventional_data_x[temporal_index][best_es] = data_x
        self.interventional_data_y[temporal_index][best_es] = data_y