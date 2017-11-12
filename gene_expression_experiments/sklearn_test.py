from data_loader import read_tp63_dataset
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def gene_expression_classifier(expr_data,
                               profile_cols=None):
    if profile_cols is None:
        profile_cols = np.array([0, 20, 40, 60, 80, 100, 120,
                                 140, 160, 180, 200, 220, 240],
                                dtype=np.float64)
        profile_cols.shape = (len(profile_cols), 1)

    for index, row in expr_data.iterrows():

        print index
        profile = np.array(row, dtype=np.float64)
        profile.shape = (len(profile), 1)

        # Calculate the variance
        profile_var = np.var(profile)

        print "Variance: ", profile_var
        print '---------------'

        # 3) First, we define the noise model:

        noisy_kern = WhiteKernel(0.001*profile_var) * RBF(length_scale=1. / 20)
        noisy_m = GaussianProcessRegressor(kernel=noisy_kern, alpha=0.999*profile_var)
        print noisy_m.kernel.theta
        print noisy_m.get_params()
        noisy_m.fit(profile_cols, profile)
        print noisy_m.kernel_.theta
        print noisy_m.kernel.theta
        print "Optimized Model (after optimization):"
        print noisy_m.get_params()

        expr_data.loc[index, 'opt_likelihood'] = noisy_m.log_marginal_likelihood()
        expr_data.loc[index, 'noisy_likelihood'] = noisy_m.log_marginal_likelihood(theta=[np.log(0.001 * profile_var),
                                                                                          np.log(.001)])

        print "=============================="

    return expr_data


tp63 = read_tp63_dataset()
tp63 = tp63.loc[tp63.index.isin(['1415918_a_at', '1416039_x_at',
                                 '1416041_at', '1416612_at'])]

print gene_expression_classifier(tp63)