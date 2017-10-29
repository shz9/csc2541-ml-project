import gpflow
import tensorflow as tf


class RationalQuadratic(gpflow.kernels.Stationary):
    """
    This class extends the Stationary kernel in GPflow and
    implements the Rational Quadratic kernel.

    Based on:
    https://github.com/GPflow/GPflow/issues/526#issuecomment-338256036

    """
    def __init__(self, input_dim, active_dims=None, alpha=1.0, lengthscales=1.0):
        gpflow.kernels.Stationary.__init__(self, input_dim=input_dim, active_dims=active_dims)
        self.alpha = gpflow.param.Param(alpha, transform=gpflow.transforms.positive)
        self.lengthscales = lengthscales

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.pow(1 + (self.square_dist(X, X2) / (2*self.alpha*(self.lengthscales ** 2))),
                                      -1 * self.alpha)
