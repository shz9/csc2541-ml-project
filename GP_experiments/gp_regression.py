import pandas as pd
from visualization.regression_plotters import *
from kernels.CustomKernels import *

# Author: Shadi Zabad
# The code below is an attempt to fit the Mauna Loa dataset using GPflow.
# I'm following the same procedures as this Scikit-learn tutorial:
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py

# Load the Mauna Loa CO2 dataset and label the columns
df = pd.read_csv('../_data/mauna-loa-atmospheric-co2.csv', sep=',', header=None)
df.columns = ['CO2Concentration', 'Time']

# Extract the X and Y components
X_all = np.array(df['Time']).reshape((len(df), 1))
Y_all = np.array(df['CO2Concentration']).reshape((len(df), 1))

# Extract the training data & extract X and Y components
train_data = df.loc[df.Time <= 1980, ['CO2Concentration', 'Time']]
X_train = np.array(train_data['Time']).reshape((len(train_data), 1))
Y_train = np.array(train_data['CO2Concentration']).reshape((len(train_data), 1))

# Use custom kernel to model the data:
custom_kernel = mauna_loa_kernel()
m = gpflow.gpr.GPR(X_train, Y_train, kern=custom_kernel)

# Fit GPR to the training data
m.optimize()

# Create 5000 X points from start to end of the dataset.
xx = np.linspace(min(df["Time"]), max(df["Time"]), 5000)[:, None]
# Predict corresponding Y points (i.e. CO2 concentrations).
gp_mean, gp_var = m.predict_y(xx)

# Send results to the plotting function.
gp_p = {
    "X": xx,
    "Y": gp_mean,
    "Var": gp_var
}

gp_plot("./_images/mauna_loa_co2_gpr.png", X_all, Y_all, gp_p)
