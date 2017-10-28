import gpflow
from RationalQuadratic import RationalQuadratic


def mauna_loa_kernel():
    """
    This kernel implementation is based on the scikit-learn tutorial:
    http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
    :return:
    """
    # Long term smooth rising trend:
    k1 = gpflow.kernels.Constant(66.0**2) * gpflow.kernels.RBF(1, lengthscales=67.0)

    # Seasonal component:
    k2 = gpflow.kernels.Constant(2.4**2) * gpflow.kernels.RBF(1, lengthscales=90.0) \
        * gpflow.kernels.PeriodicKernel(1, lengthscales=1.3, period=1.0)  # seasonal component

    # Medium term irregularity:
    k3 = gpflow.kernels.Constant(0.66**2) * RationalQuadratic(lengthscales=1.2, alpha=0.78)

    # Noise terms:
    k4 = gpflow.kernels.Constant(0.18**2) * gpflow.kernels.RBF(1, lengthscales=0.134) + \
         gpflow.kernels.White(1, variance=0.19**2)

    return k1 + k2 + k3 + k4
