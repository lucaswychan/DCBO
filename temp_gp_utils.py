from copy import deepcopy
from typing import Callable, OrderedDict, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .sequential_sampling import sequential_sample_from_SEM_hat, sequential_sample_from_true_SEM

def fit_gp(x, y, lengthscale=1.0, variance=1.0, noise_var=1.0, n_restart=10, seed: int = 0) -> Callable:
    """Fits a GP model using TensorFlow Probability instead of GPy"""
    print("Using gp_utils for fit_gp")
    if seed is not None:
        tf.random.set_seed(seed)
        
    # Ensure inputs are 2D arrays with correct shape
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        
    # Convert to tensors
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    
    # Create kernel
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=tf.Variable(variance, dtype=tf.float64),
        length_scale=tf.Variable(lengthscale, dtype=tf.float64)
    )
    
    # Create mean function
    mean_fn = lambda x: tf.zeros(tf.shape(x)[:-1], dtype=tf.float64)
    
    # Create GP model with correct parameters
    gp = tfp.distributions.GaussianProcess(
        kernel_provider=kernel,
        index_points=X,
        observation_noise_variance=0.1)
    
    # Optimize hyperparameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def objective():
        loss = -gp.log_prob(y)
        return tf.where(tf.math.is_finite(loss), loss, tf.float64.max)
    
    best_loss = float('inf')
    best_params = {
        'amplitude': variance,
        'length_scale': lengthscale
    }
    
    # Multiple restarts for optimization
    for _ in range(n_restart):
        try:
            # Random initialization
            kernel.amplitude.assign(tf.random.uniform([], 0.1, 2.0, dtype=tf.float64))
            kernel.length_scale.assign(tf.random.uniform([], 0.1, 2.0, dtype=tf.float64))
            
            # Optimize using gradient tape
            for _ in range(100):
                with tf.GradientTape() as tape:
                    loss = objective()
                
                # Get gradients
                grads = tape.gradient(loss, [kernel.amplitude, kernel.length_scale])
                
                # Check if gradients are valid
                if all(g is not None and tf.reduce_all(tf.math.is_finite(g)) for g in grads):
                    # Apply gradients
                    optimizer.apply_gradients(zip(grads, [kernel.amplitude, kernel.length_scale]))
            
            current_loss = objective().numpy()
            if np.isfinite(current_loss) and current_loss < best_loss:
                best_loss = current_loss
                best_params = {
                    'amplitude': kernel.amplitude.numpy(),
                    'length_scale': kernel.length_scale.numpy()
                }
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print(f"Optimization failed for this restart: {e}")
            continue
    
    # Set best parameters
    kernel.amplitude.assign(best_params['amplitude'])
    kernel.length_scale.assign(best_params['length_scale'])
    
    # Add predict_f method to match GPy interface
    def predict_f(X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        
        gp_pred = tfp.distributions.GaussianProcessRegressionModel(
            kernel=kernel,
            index_points=X,
            observation_index_points=x,
            observations=y,
            observation_noise_variance=tf.Variable(noise_var, dtype=tf.float64),
            mean_fn=mean_fn,
            jitter=1e-6
        )
        mean = gp_pred.mean()
        var = gp_pred.variance()
        return (
            tf.where(tf.math.is_finite(mean), mean, tf.zeros_like(mean)),
            tf.where(tf.math.is_finite(var), var, tf.ones_like(var))
        )
    
    gp.predict_f = predict_f
    return gp

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
) -> Tuple[Callable, Callable]:
    """Updates sufficient statistics using TFP implementation"""
    print("Using gp_utils for update_sufficient_statistics_hat")
    
    if dynamic:
        intervention_blanket = deepcopy(assigned_blanket)
        dynamic_sem_mean = sem_hat().dynamic(moment=0)
        dynamic_sem_var = sem_hat().dynamic(moment=1)
    else:
        intervention_blanket = deepcopy(assigned_blanket)
        assert [all(intervention_blanket[key] is None for key in intervention_blanket.keys())]
        dynamic_sem_mean = None
        dynamic_sem_var = None

    kwargs1 = {
        "static_sem": sem_hat().static(moment=0),
        "dynamic_sem": dynamic_sem_mean,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }
    
    kwargs2 = {
        "static_sem": sem_hat().static(moment=1),
        "dynamic_sem": dynamic_sem_var,
        "node_parents": node_parents,
        "timesteps": temporal_index + 1,
    }

    @tf.function
    def mean_function(x_vals):
        def get_sample(x):
            x_key = tf.strings.as_string(x)
            if x_key in mean_dict_store[temporal_index][exploration_set]:
                return mean_dict_store[temporal_index][exploration_set][x_key]
            
            for intervention_variable, xx in zip(exploration_set, tf.unstack(x)):
                intervention_blanket[intervention_variable][temporal_index] = xx.numpy()
            
            sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs1, seed=seed)
            result = sample[target_variable][temporal_index]
            mean_dict_store[temporal_index][exploration_set][x_key] = result
            return result
        
        return tf.stack([get_sample(x) for x in tf.unstack(x_vals)])

    @tf.function
    def variance_function(x_vals):
        def get_variance(x):
            x_key = tf.strings.as_string(x)
            if x_key in var_dict_store[temporal_index][exploration_set]:
                return var_dict_store[temporal_index][exploration_set][x_key]
            
            for intervention_variable, xx in zip(exploration_set, tf.unstack(x)):
                intervention_blanket[intervention_variable][temporal_index] = xx.numpy()
            
            sample = sequential_sample_from_SEM_hat(interventions=intervention_blanket, **kwargs2, seed=seed)
            result = sample[target_variable][temporal_index]
            var_dict_store[temporal_index][exploration_set][x_key] = result
            return result
        
        return tf.stack([get_variance(x) for x in tf.unstack(x_vals)])

    return mean_function, variance_function

# The update_sufficient_statistics function remains largely unchanged but uses TFP models internally

