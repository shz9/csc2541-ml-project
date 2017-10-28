import pandas as pd
from visualization.regression_plotters import *
from kernels.CustomKernels import *

# Author: Shadi Zabad
# The code below is an attempt to fit the Mauna Loa dataset using GPflow.
# I'm following the same procedures as this Scikit-learn tutorial:
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
# For some reason, the kernel I implemented isn't capturing the periodic behavior in its future projections.

df = pd.read_csv('../_data/mauna-loa-atmospheric-co2.csv', sep=',', header=None)
df.columns = ['CO2Concentration', 'Time']

X_all = np.array(df['Time']).reshape((len(df), 1))
Y_all = np.array(df['CO2Concentration']).reshape((len(df), 1))

train_data = df.loc[df.Time <= 1980, ['CO2Concentration', 'Time']]
# test_data = df.loc[df.Time > 1980, ['CO2Concentration', 'Time']]

X_train = np.array(train_data['Time']).reshape((len(train_data), 1))
Y_train = np.array(train_data['CO2Concentration']).reshape((len(train_data), 1))

gp_plot(X_train, Y_train)

custom_kernel = mauna_loa_kernel()

m = gpflow.gpr.GPR(X_train, Y_train, kern=custom_kernel)
m.optimize()

xx = np.linspace(min(df["Time"]), max(df["Time"]), 5000)[:, None]

gp_mean, gp_var = m.predict_y(xx)

gp_p = {
    "X": xx,
    "Y": gp_mean,
    "Var": gp_var
}

gp_plot(X_all, Y_all, gp_p)
