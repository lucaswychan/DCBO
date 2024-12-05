from typing import Callable
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Enable NumPy behavior in TensorFlow
tf.experimental.numpy.experimental_enable_numpy_behavior()

from dcbo.bases.root import Root
from dcbo.bayes_opt.causal_kernels import CausalRBF
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function
from dcbo.utils.sem_utils.sem_estimate import fit_arcs
from dcbo.utils.utilities import (
    convert_to_dict_of_temporal_lists,
    standard_mean_function,
    zero_variance_adjustment,
)
from tqdm import trange
from dcbo.utils.tfp_model_wrapper import TFPModelWrapper

tfk = tfp.math.psd_kernels
tfb = tfp.bijectors
tfd = tfp.distributions

class CBO(Root):
    def __init__(
        self,
        G: str,
        sem: classmethod,
        make_sem_estimator: Callable,
        observation_samples: dict,
        intervention_domain: dict,
        intervention_samples: dict,
        exploration_sets: list,
        number_of_trials: int,
        base_target_variable: str,
        ground_truth: list = None,
        estimate_sem: bool = True,
        task: str = "min",
        n_restart: int = 1,
        cost_type: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        concat: bool = False,
        optimal_assigned_blankets: dict = None,
        n_obs_t: int = None,
        hp_i_prior: bool = True,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        args = {
            "G": G,
            "sem": sem,
            "make_sem_estimator": make_sem_estimator,
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": exploration_sets,
            "estimate_sem": estimate_sem,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "use_mc": use_mc,
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        super().__init__(**args)

        self.concat = concat
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.n_obs_t = n_obs_t
        self.hp_i_prior = hp_i_prior
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        
        # Fit Gaussian processes to emissions
        self.sem_emit_fncs = fit_arcs(self.G, self.observational_samples, emissions=True)
        # Convert observational samples to dict of temporal lists
        self.observational_samples = convert_to_dict_of_temporal_lists(self.observational_samples)

    def _update_bo_model(
        self, temporal_index: int, exploration_set: tuple, alpha: float = 2, beta: float = 0.5,
    ) -> None:
        """Update Bayesian optimization model using TFP"""
        
        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        input_dim = len(exploration_set)
        # Convert to numpy first, then to tensor to ensure compatibility
        X = tf.convert_to_tensor(np.array(self.interventional_data_x[temporal_index][exploration_set]), dtype=tf.float32)
        Y = tf.convert_to_tensor(np.array(self.interventional_data_y[temporal_index][exploration_set]), dtype=tf.float32)

        # Create mean function wrapper with numpy compatibility
        mean_fn = lambda x: tf.convert_to_tensor(
            np.array(self.mean_function[temporal_index][exploration_set](x)), 
            dtype=tf.float32
        )

        # Set random seed for reproducibility
        tf.random.set_seed(self.seed)

        if temporal_index > 0 and isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
            # Standard RBF kernel (same as GPy's RBF)
            kernel = tfk.ExponentiatedQuadratic(
                amplitude=tf.Variable(1.0, dtype=tf.float32),
                length_scale=tf.Variable(1.0, dtype=tf.float32)
            )
        else:
            # Causal kernel with variance adjustment
            kernel = CausalRBF(
                input_dim=input_dim,
                variance_adjustment=lambda x: tf.convert_to_tensor(
                    np.array(self.variance_function[temporal_index][exploration_set](x)),
                    dtype=tf.float32
                ),
                lengthscale=1.0,
                variance=1.0,
                ARD=False
            )

        # Implement hyperparameter prior (equivalent to GPy's Gamma prior)
        if self.hp_i_prior:
            constrain_positive = tfb.Softplus()
            kernel.variance = tf.Variable(
                constrain_positive.forward(1.0), 
                constraint=constrain_positive
            )

        # Optimize model (equivalent to GPy's optimize)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        def loss_fn():
            gp = tfd.GaussianProcess(
                kernel=kernel,
                index_points=X,
                observation_noise_variance=1e-5,
                mean_fn=mean_fn
            )
            return -gp.log_prob(Y)  # Negative log likelihood

        # Train for 100 iterations (similar to GPy's optimization)
        for _ in range(100):
            with tf.GradientTape() as tape:
                loss = loss_fn()
            
            # Get gradients - use the correct attribute names
            if isinstance(kernel, tfk.ExponentiatedQuadratic):
                grads = tape.gradient(loss, [kernel.amplitude, kernel.length_scale])
                trainable_vars = [kernel.amplitude, kernel.length_scale]
            else:
                grads = tape.gradient(loss, [kernel.variance, kernel.lengthscale])
                trainable_vars = [kernel.variance, kernel.lengthscale]
            
            # Apply gradients
            optimizer.apply_gradients(zip(grads, trainable_vars))

        # Create and store model wrapper
        self.bo_model[temporal_index][exploration_set] = TFPModelWrapper(
            kernel=kernel,
            mean_function=mean_fn,
            noise_variance=1e-5,
            X=X,
            Y=Y
        )

        # Safe optimization check (same as GPy version)
        self._safe_optimization(temporal_index, exploration_set)

    def _safe_optimization(self, temporal_index, exploration_set):
        """Safely optimize the model with multiple restarts (equivalent to GPy's optimize_restarts)"""
        if self.n_restart > 1:
            best_nlml = float("inf")
            best_model = None
            
            for _ in range(self.n_restart):
                # Random initialization (same as GPy)
                kernel = CausalRBF(
                    input_dim=len(exploration_set),
                    variance_adjustment=self.variance_function[temporal_index][exploration_set],
                    lengthscale=tf.random.uniform([], 0.1, 2.0),
                    variance=tf.random.uniform([], 0.1, 2.0),
                    ARD=False
                )
                
                X = tf.convert_to_tensor(self.interventional_data_x[temporal_index][exploration_set], dtype=tf.float32)
                Y = tf.convert_to_tensor(self.interventional_data_y[temporal_index][exploration_set], dtype=tf.float32)
                mean_fn = lambda x: tf.convert_to_tensor(
                    self.mean_function[temporal_index][exploration_set](x), 
                    dtype=tf.float32
                )
                
                # Optimize
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                
                def loss_fn():
                    gp = tfd.GaussianProcess(
                        kernel=kernel,
                        index_points=X,
                        observation_noise_variance=1e-5,
                        mean_fn=mean_fn
                    )
                    return -gp.log_prob(Y)
                
                for _ in range(100):
                    with tf.GradientTape() as tape:
                        loss = loss_fn()
                    
                    # Get gradients
                    grads = tape.gradient(loss, [kernel.amplitude, kernel.length_scale])
                    # Apply gradients
                    optimizer.apply_gradients(zip(grads, [kernel.amplitude, kernel.length_scale]))
                
                # Keep best model (same as GPy)
                current_nlml = loss_fn().numpy()
                if current_nlml < best_nlml:
                    best_nlml = current_nlml
                    best_model = TFPModelWrapper(
                        kernel=kernel,
                        mean_function=mean_fn,
                        noise_variance=1e-5,
                        X=X,
                        Y=Y
                    )
            
            if best_model is not None:
                self.bo_model[temporal_index][exploration_set] = best_model

    def _update_sem_emit_fncs(self, t: int) -> None:
        """Update emission functions using TFP"""
        
        for pa in self.sem_emit_fncs[t]:
            xx, yy = self._get_sem_emit_obs(t, pa)
            if xx and yy:
                X = tf.convert_to_tensor(xx, dtype=tf.float32)
                Y = tf.convert_to_tensor(yy, dtype=tf.float32)
                
                kernel = tfk.ExponentiatedQuadratic()
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                
                def loss_fn():
                    gp = tfd.GaussianProcess(
                        kernel=kernel,
                        index_points=X,
                        observation_noise_variance=1e-5
                    )
                    return -gp.log_prob(Y)
                
                for _ in range(100):
                    with tf.GradientTape() as tape:
                        loss = loss_fn()
                    
                    # Get gradients
                    grads = tape.gradient(loss, kernel.trainable_variables)
                    # Apply gradients
                    optimizer.apply_gradients(zip(grads, kernel.trainable_variables))
                
                self.sem_emit_fncs[t][pa] = TFPModelWrapper(
                    kernel=kernel,
                    mean_function=lambda x: tf.zeros_like(x[..., 0]),
                    noise_variance=1e-5,
                    X=X,
                    Y=Y
                )

    def _get_assigned_blanket(self, temporal_index):
        """Get the assigned blanket for the current temporal index"""
        if temporal_index > 0:
            if self.optimal_assigned_blankets is not None:
                assigned_blanket = self.optimal_assigned_blankets[temporal_index]
            else:
                assigned_blanket = self.assigned_blanket
        else:
            assigned_blanket = self.assigned_blanket
        return assigned_blanket

    def _update_interventional_data(self, temporal_index):
        """Update interventional data based on temporal index"""
        if temporal_index > 0 and self.concat:
            for var in self.interventional_data_x[0].keys():
                self.interventional_data_x[temporal_index][var] = self.interventional_data_x[temporal_index - 1][var]
                self.interventional_data_y[temporal_index][var] = self.interventional_data_y[temporal_index - 1][var]

    def run(self):
        """Main optimization loop"""
        if self.debug_mode:
            assert self.ground_truth is not None, "Provide ground truth to plot surrogate models"

        # Walk through the graph temporally
        for temporal_index in trange(self.T, desc="Time index"):
            # Get current target variable
            target = self.all_target_variables[temporal_index]
            _, target_temporal_index = target.split("_")
            assert int(target_temporal_index) == temporal_index
            best_es = self.best_initial_es

            # Update data
            self._update_observational_data(temporal_index=temporal_index)
            self._update_interventional_data(temporal_index=temporal_index)

            # Online updates
            if temporal_index > 0 and (self.online or isinstance(self.n_obs_t, list)):
                self._update_sem_emit_fncs(temporal_index)

            # Get blanket for computing y_new
            assigned_blanket = self._get_assigned_blanket(temporal_index)

            for it in range(self.number_of_trials):
                if it == 0:
                    self.trial_type[temporal_index].append("o")  # 'o'bserve
                    sem_hat = self.make_sem_hat(G=self.G, emission_fncs=self.sem_emit_fncs)

                    # Update prior using observational data
                    self._update_sufficient_statistics(
                        target=target,
                        temporal_index=temporal_index,
                        dynamic=False,
                        assigned_blanket=self.empty_intervention_blanket,
                        updated_sem=sem_hat,
                    )
                    # Update optimization parameters
                    self._update_opt_params(it, temporal_index, best_es)

                else:
                    # Update surrogate models if needed
                    if self.trial_type[temporal_index][-1] == "o":
                        for es in self.exploration_sets:
                            if (
                                self.interventional_data_x[temporal_index][es] is not None
                                and self.interventional_data_y[temporal_index][es] is not None
                            ):
                                self._update_bo_model(temporal_index, es)

                    # Run computations for this trial
                    self._per_trial_computations(temporal_index, it, target, assigned_blanket)

            # Post-optimization assignments
            self._post_optimisation_assignments(target, temporal_index)

    def _evaluate_acquisition_functions(self, temporal_index, current_best_global_target, it):
        """Evaluate acquisition functions for all exploration sets"""
        
        for es in self.exploration_sets:
            # Get BO model if available
            if (
                self.interventional_data_x[temporal_index][es] is not None
                and self.interventional_data_y[temporal_index][es] is not None
            ):
                bo_model = self.bo_model[temporal_index][es]
            else:
                bo_model = None
                if isinstance(self.n_obs_t, list) and self.n_obs_t[temporal_index] == 1:
                    self.mean_function[temporal_index][es] = standard_mean_function
                    self.variance_function[temporal_index][es] = zero_variance_adjustment

            # Set seed for anchor points
            seed_to_pass = None
            if self.seed_anchor_points is not None:
                seed_to_pass = int(self.seed_anchor_points * (temporal_index + 1) * it)

            # Get the parameter space
            parameter_space = self.intervention_exploration_domain[es]

            try:
                self.y_acquired[es], self.corresponding_x[es] = evaluate_acquisition_function(
                    parameter_intervention_domain=parameter_space,  # Pass the ParameterSpace object directly
                    bo_model=bo_model,
                    mean_function=self.mean_function[temporal_index][es],
                    variance_function=self.variance_function[temporal_index][es],
                    optimal_target_value_at_current_time=current_best_global_target,
                    exploration_set=es,
                    cost_functions=self.cost_functions,
                    task=self.task,
                    base_target=self.base_target_variable,
                    dynamic=False,
                    causal_prior=True,
                    temporal_index=temporal_index,
                    previous_variance=1.0,
                    num_anchor_points=self.num_anchor_points,
                    sample_anchor_points=self.sample_anchor_points,
                    seed_anchor_points=seed_to_pass,
                )
            except Exception as e:
                print(f"Error in acquisition function evaluation: {str(e)}")
                print(f"Parameter space bounds: {parameter_space.get_bounds()}")
                print(f"Number of anchor points: {self.num_anchor_points}")
                print(f"Exploration set: {es}")
                raise e