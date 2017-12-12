# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.datasets import fetch_mldata

# get data
data = fetch_mldata('mauna-loa-atmospheric-co2').data
x = data[:, [1]]
y = data[:, 0]
x_train, y_train = x[:int(0.7*len(x))], y[:int(0.7*len(y))]
x_test, y_test = x[int(0.7*len(x)):], y[int(0.7*len(y)):]

# Kernel from [1]
# it was implemented by Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
k1 = 50.0**2 * RBF(length_scale=50.0)  # linear trend
k2 = 2.0**2 * RBF(length_scale=100.0) \
    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                     periodicity_bounds="fixed")  # oscillations
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf)) # noise
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)

# fit GP
gp.fit(x_train, y_train)
# produce vector of values on x-axis
X_ = np.linspace(x.min(), x.max(), 1000)[:, np.newaxis]
xticks = np.linspace(0, len(y), 1000)[:, np.newaxis]
# predict
y_pred, y_std = gp.predict(X_, return_std=True)

# figure
plt.figure(num=None, figsize=(4,4), dpi=300)
plt.rcParams.update({'font.size': 6})
plt.plot(list(range(len(x))), y, c='b', label='Actual', linewidth=1) # true data
plt.plot(xticks, y_pred, c='r', label='GP ($K_1$)', linewidth=1) # predictions
plt.fill_between(xticks[:, 0], y_pred-2*y_std, y_pred+2*y_std,
                 alpha=0.3, color='#C0C0C0') # confidence intervals
plt.ylim(310, 380)
lo,hi = plt.ylim()
plt.plot([list(range(len(x)))[int(0.7*len(x))], list(range(len(x)))[int(0.7*len(x))]],[lo,hi],'k--')
plt.xlim(xticks.min(), xticks.max())
plt.xlabel("Months")
plt.ylabel("CO2 Concentration (PPM)")
plt.rcParams['axes.ymargin'] = 0
plt.tight_layout()
plt.show()

# calculate MSE
#y_pred, y_std = gp.predict(x, return_std=True)
#mse = np.mean((y_test-y_pred[int(0.7*len(x)):])**2)

# Reference
# [1] Carl Edward Rasmussen and Christopher K.I. Williams. Gaussian Processes for machine learning, volume 14. 2006.