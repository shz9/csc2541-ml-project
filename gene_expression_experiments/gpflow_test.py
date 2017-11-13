from data_loader import read_tp63_dataset
import numpy as np
import gpflow


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

        noisy_kern = gpflow.kernels.RBF(1, lengthscales=np.log(1. / 1000.0),
                                        variance=np.log(.001 * profile_var)) #+ \
                     #gpflow.kernels.White(1, variance=np.log(0.001 * profile_var))

        #noisy_kern.lengthscales.transform = gpflow.transforms.Exp()
        #noisy_kern.variance.transform = gpflow.transforms.Exp()

        noisy_m = gpflow.gpr.GPR(X=profile_cols,
                                 Y=profile,
                                 kern=noisy_kern)

        #noisy_m.kern.lengthscales.fixed = False
        #noisy_m.kern.variance.fixed = True
        # noisy_m.likelihood.variance.fixed = True
        # noisy_m.kern.white.variance.fixed = True

        #noisy_m.optimize()

        print noisy_m
        print "^^^^^^^^"

        # --------------------------
        opt_kern = gpflow.kernels.RBF(1, lengthscales=1.,
                                      variance=1.) #+ \
                   #gpflow.kernels.White(1, variance=1.)

        opt_m = gpflow.gpr.GPR(X=profile_cols,
                               Y=profile,
                               kern=opt_kern)

        print "Optimized Model:"
        print opt_m
        print '--------------'

        opt_m.optimize(method='CG')

        print "Optimized Model (after optimization):"
        print opt_m

        expr_data.loc[index, 'noisy_likelihood'] = noisy_m.compute_log_likelihood()
        expr_data.loc[index, 'opt_likelihood'] = opt_m.compute_log_likelihood()

        print "=============================="

    return expr_data


tp63 = read_tp63_dataset()
tp63 = tp63.loc[tp63.index.isin(['1415918_a_at', '1416039_x_at',
                                 '1416041_at', '1416612_at'])]

print gene_expression_classifier(tp63)

