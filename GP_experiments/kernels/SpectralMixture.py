import gpflow
import tensorflow as tf


class SpectralMixture(gpflow.kernels.Kern):

    def __init__(self):
        gpflow.kernels.Kern.__init__(self, input_dim=1, active_dims=[0])
        self.variance = gpflow.param.Param(1.0, transform=gpflow.transforms.positive)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        pass

    def Kdiag(self, X):
        # return self.variance * tf.reshape(X, (-1,))
        pass
