from copy import deepcopy
from typing import Callable, OrderedDict, Tuple

import numpy as np
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

import tensorflow as tf
import tensorflow_probability as tfp

from .sequential_sampling import sequential_sample_from_SEM_hat, sequential_sample_from_true_SEM


def update_sufficient_statistics_hat(
    temporal_index: int,
    target_variable: str,
    exploration_set: tuple,
    sem_hat: OrderedDict,
    node_parents: Callable,
    dynamic: bool,
    assigned_blanket: dict,
    mean_dict_store: dict,
    var_dict_store: dict,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the mean and variance functions (priors) on our causal effects given the current exploration set.

    Parameters
    ----------
    temporal_index : int
        The current time index in the causal Bayesian network.
    target_variable : str
        The current target variable e.g Y_1
    exploration_set : tuple
        The current exploration set
    sem_hat : OrderedDict
        Contains our estimated SEMs
    node_parents : Callable
        Function with returns parents of the passed argument at the given time-slice
    dynamic : bool
        Tells the method to use horizontal information or not
    assigned_blanket : dict
        The assigned blanket thus far (i.e. up until the temporal index)
    mean_dict_store : dict
        Stores the updated mean function for this time index and exploration set
    var_dict_store : dict
        Stores the updated variance function for this time index and exploration set
    seed : int, optional
        The random seet, by default 1

    Returns
    -------
    Tuple
        Returns the updated mean and variance function
    """

    if dynamic:
        # This relies on the correct blanket being passed from outside this function.
        intervention_blanket = deepcopy(assigned_blanket)
        dynamic_sem_mean = sem_hat().dynamic(moment=0)
        dynamic_sem_var = sem_hat().dynamic(moment=1)
    else:
        # static: no backward dependency on past targets or interventions
        intervention_blanket = deepcopy(
            assigned_blanket
        )  # This is empty if passed correctly. This relies on the correct blanket being passed from outside this function.
        assert [all(intervention_blanket[key] is None for key in intervention_blanket.keys())]
        dynamic_sem_mean = None  # CBO does not have horizontal information hence gets static model every time
        dynamic_sem_var = None  # CBO does not have horizontal information hence gets static model every time

    #  Mean vars
    kwargs1 = {
        "static_sem": sem_hat().static(moment=0),  # Get the mean
        "dynamic_sem": dynamic_sem_mean,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }
    #  Variance vars
    kwargs2 = {
        "static_sem": sem_hat().static(moment=1),  # Gets the variance
        "dynamic_sem": dynamic_sem_var,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }

    def mean_function_internal(x_vals, mean_dict_store) -> np.ndarray:
        samples = []
        for x in x_vals:
            # Check if it has already been computed
            if str(x) in mean_dict_store[temporal_index][exploration_set]:
                samples.append(mean_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx

                # TODO: parallelise all sampling functions, this is much too slow [GPyTorch] -- see https://docs.gpytorch.ai/en/v1.5.0/examples/08_Advanced_Usage/Simple_Batch_Mode_GP_Regression.html#Setting-up-the-model
                sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs1, seed=seed)
                samples.append(sample[target_variable][temporal_index])
                mean_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]
        return np.vstack(samples)

    def mean_function(x_vals) -> np.ndarray:
        return mean_function_internal(x_vals, mean_dict_store)

    def variance_function_internal(x_vals, var_dict_store):
        out = []
        for x in x_vals:
            # Check if it is already computed
            if str(x) in var_dict_store[temporal_index][exploration_set]:
                out.append(var_dict_store[temporal_index][exploration_set][str(x)])
            else:
                # Otherwise compute it and store it
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][temporal_index] = xx
                # TODO: parallelise all sampling functions, this is much too slow
                sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs2, seed=seed)
                out.append(sample[target_variable][temporal_index])
                var_dict_store[temporal_index][exploration_set][str(x)] = sample[target_variable][temporal_index]
        return np.vstack(out)

    def variance_function(x_vals) -> np.ndarray:
        return variance_function_internal(x_vals, var_dict_store)

    return mean_function, variance_function


def update_sufficient_statistics(
    temporal_index: int,
    exploration_set: tuple,
    time_slice_children: dict,
    initial_sem: dict,
    sem: dict,
    dynamic: bool,
    assigned_blanket: dict,
) -> Tuple[np.ndarray, np.ndarray]:

    if dynamic:
        # This relies on the correct blanket being passed from outside this function.
        intervention_blanket = deepcopy(assigned_blanket)
    else:
        # static: no backward dependency on past targets or interventions
        # This relies on the correct blanket being passed from outside this function.
        # This is empty if passed correctly.
        intervention_blanket = deepcopy(assigned_blanket)

        assert [all(intervention_blanket[key] is None for key in intervention_blanket.keys())]

    if len(exploration_set) == 1:
        #  Check which variable is currently being intervened upon
        intervention_variable = exploration_set[0]  # Input variable to SEM
        child_var = time_slice_children[intervention_variable]

        def mean_function(x_vals):
            out = []
            for x in x_vals:
                intervention_blanket[intervention_variable][temporal_index] = x
                sample = sequential_sample_from_true_SEM(
                    initial_sem, sem, temporal_index + 1, interventions=intervention_blanket,
                )
                out.append(sample[child_var][temporal_index])
            return np.vstack(out)

        def variance_function(x_vals):
            return np.zeros(x_vals.shape)

    else:

        def mean_function(x_vals):
            out = []
            for x in x_vals:
                for i, inter_var in enumerate(exploration_set):
                    intervention_blanket[inter_var][temporal_index] = x[i]
                    sample = sequential_sample_from_true_SEM(
                        initial_sem, sem, temporal_index + 1, interventions=intervention_blanket,
                    )
                    out.append(sample[child_var][temporal_index])
            return np.vstack(out)

        def variance_function(x_vals):
            return np.zeros(x_vals.shape)

    return mean_function, variance_function


def fit_gp(x, y, lengthscale=1.0, variance=1.0, noise_var=1.0, ard=False, n_restart=10, seed: int = 0) -> Callable:
    # Set random seed
    tf.random.set_seed(seed)
    
    # Convert inputs to tensors and reshape y if needed
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    if len(y.shape) == 1:
        y = tf.reshape(y, (-1, 1))
    
    # Initialize kernel parameters
    if ard:
        # For ARD, we need a lengthscale per input dimension
        initial_lengthscale = tf.ones([x.shape[1]], dtype=tf.float32) * lengthscale
    else:
        initial_lengthscale = tf.constant(lengthscale, dtype=tf.float32)
    
    amplitude = tf.Variable(variance, dtype=tf.float32)
    length_scale = tf.Variable(initial_lengthscale, dtype=tf.float32)
    observation_noise_variance = tf.Variable(noise_var, dtype=tf.float32)
    
    def build_gp():
        # Define the kernel
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude,
            length_scale=length_scale
        )
        
        # Define GP model
        gp = tfp.distributions.GaussianProcessRegressionModel(
            kernel=kernel,
            index_points=x,
            observation_index_points=x,
            observations=y,
            observation_noise_variance=observation_noise_variance,
            jitter=1e-6
        )
        return gp
    
    # Function to compute negative åog likelihood
    def neg_log_likelihood():
        gp = build_gp()
        return -gp.log_prob(y)
    
    # Optimization
    def optimize_one_restart():
        # Reset variables
        amplitude.assign(variance)
        if ard:
            length_scale.assign(tf.ones([x.shape[1]], dtype=tf.float32) * lengthscale)
        else:
            length_scale.assign(lengthscale)
        observation_noise_variance.assign(noise_var)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for _ in range(100):  # Max iterations
            with tf.GradientTape() as tape:
                loss = neg_log_likelihood()
            
            gradients = tape.gradient(loss, [amplitude, length_scale, observation_noise_variance])
            optimizer.apply_gradients(zip(gradients, [amplitude, length_scale, observation_noise_variance]))
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_loss.numpy()
    
    # Multiple restarts
    best_loss = float('inf')
    best_params = None
    
    for i in range(n_restart):
        tf.random.set_seed(seed + i + 1)
        current_loss = optimize_one_restart()
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = {
                'amplitude': amplitude.numpy(),
                'length_scale': length_scale.numpy(),
                'noise_var': observation_noise_variance.numpy()
            }
    
    # Set the best parameters
    amplitude.assign(best_params['amplitude'])
    length_scale.assign(best_params['length_scale'])
    observation_noise_variance.assign(best_params['noise_var'])
    
    # Return prediction function
    def predict(x_new):
        x_new = tf.convert_to_tensor(x_new, dtype=tf.float32)
        
        # Create kernel with optimized parameters
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude,
            length_scale=length_scale
        )
        
        # Compute kernel matrices
        k_xx = kernel.matrix(x, x) + observation_noise_variance * tf.eye(tf.shape(x)[0], dtype=tf.float32)
        k_xx_inv = tf.linalg.inv(k_xx)
        k_x_new_x = kernel.matrix(x_new, x)
        k_x_new_x_new = kernel.matrix(x_new, x_new)
        
        # Compute posterior mean and variance
        mean = tf.matmul(k_x_new_x, tf.matmul(k_xx_inv, y))
        var = k_x_new_x_new - tf.matmul(k_x_new_x, tf.matmul(k_xx_inv, tf.transpose(k_x_new_x)))
        var = tf.linalg.diag_part(var)[:, None] + observation_noise_variance
        
        return mean.numpy(), var.numpy()
    
    return predict