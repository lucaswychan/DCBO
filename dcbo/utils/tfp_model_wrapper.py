import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class TFPModelWrapper:
    def __init__(self, kernel, mean_function, noise_variance, X, Y):
        self.kernel = kernel
        self.mean_function = mean_function
        self.noise_variance = noise_variance
        self.X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        
    def predict(self, X_new):
        """Make predictions with numpy compatibility"""
        X_new = np.asarray(X_new)
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)
        X_new = tf.convert_to_tensor(X_new, dtype=tf.float32)
        
        # Calculate kernel matrices
        k_xx = self.kernel.matrix(self.X, self.X) + tf.eye(tf.shape(self.X)[0]) * self.noise_variance
        k_xxn = self.kernel.matrix(self.X, X_new)
        k_xnxn = self.kernel.matrix(X_new, X_new)
        
        # Calculate posterior mean and variance
        k_xx_inv = tf.linalg.inv(k_xx)
        mean = (
            self.mean_function(X_new) + 
            tf.matmul(tf.matmul(k_xxn, k_xx_inv, transpose_a=True),
                     self.Y - self.mean_function(self.X))
        )
        
        var = (
            k_xnxn - 
            tf.matmul(tf.matmul(k_xxn, k_xx_inv, transpose_a=True), k_xxn)
        )
        
        # Convert back to numpy for compatibility
        return mean.numpy(), var.numpy() 