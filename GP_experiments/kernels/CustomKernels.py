import gpflow
from RationalQuadratic import RationalQuadratic


def mauna_loa_kernel():
    """
    This kernel implementation is based on the scikit-learn tutorial:
    http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
    """
    # Long term smooth rising trend:
    k1 = gpflow.kernels.Constant(1, 66.0 ** 2) * gpflow.kernels.RBF(1, lengthscales=67.0)

    # Seasonal component:
    k2 = gpflow.kernels.Constant(1, 2.4**2) * gpflow.kernels.RBF(1, lengthscales=90.0) \
        * gpflow.kernels.PeriodicKernel(1, lengthscales=1.3, period=1.0)

    # Medium term irregularity:
    k3 = gpflow.kernels.Constant(1, 0.66**2) * RationalQuadratic(1, lengthscales=1.2, alpha=0.78)

    # Noise terms:
    k4 = gpflow.kernels.Constant(1, 0.18**2) * gpflow.kernels.RBF(1, lengthscales=0.134) + \
         gpflow.kernels.White(1, variance=0.19**2)

    return k1 + k2 + k3 + k4


def spectral_mixture():
    """
    Based on this implementation by James Hensman:
    https://github.com/GPflow/GPflow/issues/312#issuecomment-272425105

    NOTE: This doesn't work! It's not capturing the seasonal periodic component.
    """

    return (gpflow.kernels.Constant(1) * gpflow.kernels.RBF(1) *
            gpflow.kernels.Cosine(1)) + \
           (gpflow.kernels.Constant(1) * gpflow.kernels.RBF(1) *
            gpflow.kernels.Cosine(1)) + \
           (gpflow.kernels.Constant(1) * gpflow.kernels.RBF(1) *
            gpflow.kernels.Cosine(1))
