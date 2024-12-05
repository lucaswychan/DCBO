from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dcbo.bayes_opt.causal_kernels import CausalRBF
from dcbo.bayes_opt.intervention_computations import evaluate_acquisition_function
from dcbo.utils.gp_utils import fit_gp, sequential_sample_from_SEM_hat
from dcbo.utils.sem_utils.sem_estimate import fit_arcs
from dcbo.utils.sequential_intervention_functions import make_sequential_intervention_dict
from .root import Root

class ModelWrapper:
    """Base class for model wrappers to standardize the interface"""
    
    def predict(self, X):
        """Returns mean and variance of predictions"""
        raise NotImplementedError
    
    def set_data(self, X, Y):
        """Updates model with new data"""
        raise NotImplementedError
    
    def optimize(self):
        """Optimizes model hyperparameters"""
        raise NotImplementedError

class TFPModelWrapper(ModelWrapper):
    """TensorFlow Probability GP model wrapper"""
    
    def __init__(self, model=None):
        self.model = model
        print("Initialized TFP Model Wrapper")
        
    def predict(self, X):
        """Returns mean and variance of predictions"""
        if self.model is None:
            return np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1))
        
        X = tf.convert_to_tensor(X, dtype=tf.float64)
        mean, var = self.model(X)
        return mean.numpy().reshape(-1, 1), var.numpy().reshape(-1, 1)
    
    def set_data(self, X, Y):
        """Updates model with new data"""
        X = np.asarray(X)
        Y = np.asarray(Y)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        self.model = fit_gp(X, Y)
        
    def optimize(self):
        """Optimizes model hyperparameters"""
        if self.model is not None:
            # Optimization is handled in fit_gp
            pass

class BaseClassDCBO(Root):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Using TFP implementation of BaseClassDCBO")
        self.sem_emit_fncs = fit_arcs(self.G, self.observational_samples, emissions=True)
        self.sem_trans_fncs = fit_arcs(self.G, self.observational_samples, emissions=False)

    def _update_bo_model(
        self,
        temporal_index: int,
        exploration_set: tuple,
        noise_var: float = 1e-5,
        prior_var: float = 1.0,
        prior_lengthscale: float = 1.0,
    ) -> None:
        assert self.interventional_data_x[temporal_index][exploration_set] is not None
        assert self.interventional_data_y[temporal_index][exploration_set] is not None

        X = self.interventional_data_x[temporal_index][exploration_set]
        Y = self.interventional_data_y[temporal_index][exploration_set]

        if not self.bo_model[temporal_index][exploration_set]:
            # Create new model
            old_seed = np.random.get_state()
            np.random.seed(self.seed)
            
            # Initialize model wrapper
            self.bo_model[temporal_index][exploration_set] = TFPModelWrapper()
            
            np.random.set_state(old_seed)

        # Update data and optimize
        self.bo_model[temporal_index][exploration_set].set_data(X=X, Y=Y)
        self._safe_optimization(temporal_index, exploration_set)

    def _safe_optimization(self, temporal_index, exploration_set, bound_var=1e-02, bound_len=20.0) -> None:
        """Ensures kernel parameters stay within reasonable bounds"""
        if self.bo_model[temporal_index][exploration_set].model is not None:
            model = self.bo_model[temporal_index][exploration_set].model
            kernel = model.kernel
            
            # Clip kernel parameters to safe values using TF operations
            amplitude = tf.clip_by_value(kernel.amplitude, bound_var, float('inf'))
            length_scale = tf.clip_by_value(kernel.length_scale, 0.0, bound_len)
            
            # Update kernel parameters
            kernel.amplitude.assign(amplitude)
            kernel.length_scale.assign(length_scale)

    def _get_interventional_hp(self, temporal_index, exploration_set, prior_var, prior_lengthscale):
        """Gets hyperparameters for interventional model"""
        if temporal_index > 0 and self.transfer_hp_i:
            if self.bo_model[temporal_index][exploration_set] is None:
                if self.bo_model[temporal_index - 1][exploration_set] is not None:
                    variance = self.bo_model[temporal_index - 1][exploration_set].model.kernel.amplitude.numpy()
                    lengthscale = self.bo_model[temporal_index - 1][exploration_set].model.kernel.length_scale.numpy()
                else:
                    variance = prior_var
                    lengthscale = prior_lengthscale
            else:
                variance = self.bo_model[temporal_index][exploration_set].model.kernel.amplitude.numpy()
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kernel.length_scale.numpy()
        else:
            if self.bo_model[temporal_index][exploration_set] is not None:
                variance = self.bo_model[temporal_index][exploration_set].model.kernel.amplitude.numpy()
                lengthscale = self.bo_model[temporal_index][exploration_set].model.kernel.length_scale.numpy()
            else:
                variance = prior_var
                lengthscale = prior_lengthscale

        return variance, lengthscale
