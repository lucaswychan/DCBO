import tensorflow as tf
import tensorflow_probability as tfp

class CausalRBF(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    def __init__(
        self,
        input_dim: int,
        variance_adjustment,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        ARD: bool = False,
        name: str = "CausalRBF"
    ):
        super().__init__(
            feature_ndims=1,
            dtype=tf.float32,
            name=name
        )
        self.input_dim = input_dim
        self.variance_adjustment = variance_adjustment
        self.lengthscale = tf.Variable(lengthscale, dtype=tf.float32)
        self.variance = tf.Variable(variance, dtype=tf.float32)
        self.ARD = ARD

    def _apply(self, x1, x2, example_ndims=0):
        """Apply kernel with numpy compatibility"""
        # Convert inputs to tensors if they aren't already
        x1 = tf.convert_to_tensor(x1, dtype=tf.float32)
        x2 = tf.convert_to_tensor(x2, dtype=tf.float32)
        
        # Compute RBF kernel
        scaled_diff = (x1[..., tf.newaxis, :] - x2[..., tf.newaxis, :, :]) / self.lengthscale
        squared_dist = tf.reduce_sum(tf.square(scaled_diff), axis=-1)
        rbf = self.variance * tf.exp(-0.5 * squared_dist)
        
        # Apply variance adjustment with numpy compatibility
        adjusted_variance = tf.convert_to_tensor(
            self.variance_adjustment(x1.numpy()), 
            dtype=tf.float32
        )
        return rbf * adjusted_variance[..., tf.newaxis]
