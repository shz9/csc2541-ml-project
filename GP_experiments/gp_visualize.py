from kernels.CustomKernels import *
from visualization.kernel_plotters import *


kf_test = [
    {
        "Function": gpflow.kernels.Matern12(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.Matern32(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.Matern52(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.RBF(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.Constant(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.Linear(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.Cosine(1),
        "Name": None
    },
    {
        "Function": gpflow.kernels.PeriodicKernel(1),
        "Name": None
    },
    {
        "Function": mauna_loa_kernel(),
        "Name": "CO2 Custom Kernel"
    }
]

plot_kernels(kf_test, 3, 3)
