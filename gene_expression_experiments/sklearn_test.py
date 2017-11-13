"""
Author: Shadi Zabad
Date: November 2017

Attempting to replicate the logic of Kalaitzis & Lawrence 2011 in python.
Work in progress...

"""

from data_loader import read_tp63_dataset
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from multiprocessing.pool import ThreadPool
import time


def gene_expression_classifier(expr_data,
                               profile_cols=None):
    """
    The logic below is based on Kalaitzis & Lawrence 2011.
    The implementation seems to work and the results that I'm
    getting are close to what their R script is reporting, though
    there are small discrepancies in the numbers.

    :param expr_data: The expression data as a pandas dataframe
    :param profile_cols: The profile column names (see example below).
    :return: Return updated dataframe with likelihoods computed and compared.
    """

    if profile_cols is None:
        profile_cols = np.array([0, 20, 40, 60, 80, 100, 120,
                                 140, 160, 180, 200, 220, 240],
                                dtype=np.float64)
        profile_cols.shape = (len(profile_cols), 1)

    for index, row in expr_data.iterrows():

        profile = np.array(row, dtype=np.float64)
        profile.shape = (len(profile), 1)

        # Calculate the variance in the profile
        profile_var = np.var(profile)

        # Create noisy GP
        noisy_kern = .001 * profile_var * RBF(length_scale=1. / 1000) + WhiteKernel(.999 * profile_var)
        noisy_m = GaussianProcessRegressor(kernel=noisy_kern,
                                           n_restarts_optimizer=10)

        # Fit GP to data
        noisy_m.fit(profile_cols, profile)

        # Compute log marginal likelihood of fitted GP
        expr_data.loc[index, 'opt_likelihood'] = noisy_m.log_marginal_likelihood()
        # Compute log marginal likelihood of hyperparameters that correspond to noise.
        expr_data.loc[index, 'noisy_likelihood'] = noisy_m.log_marginal_likelihood(theta=[
            np.log(.999 * profile_var),
            np.log(.05),
            np.log(.001 * profile_var)
        ])

        expr_data.loc[index, 'likelihood_ratio'] = expr_data.loc[index, 'opt_likelihood'] - \
                                                   expr_data.loc[index, 'noisy_likelihood']

    return expr_data


def process_df_in_parallel(df, func, rpp=50, num_threads=30):
    """
    This function splits a pandas dataframe to multiple smaller
    dataframes and applies the function `func` to them in parallel
    and then concatenates the results at the end.

    :param df: Dataframe to process in parallel
    :param func: A function to apply to the partitioned dataframes
    :param rpp: Rows per partition (int)
    :param num_threads: Number of threads to run in parallel
    :return: Processed dataframe
    """

    part_df = [df.iloc[i:i + rpp, ] for i in range(0, len(df), rpp)]

    t_pool = ThreadPool(num_threads)
    df = pd.concat(t_pool.map(func, part_df))

    return df


tp63 = read_tp63_dataset()

tp63 = tp63.iloc[:100, :]

start = time.time()

gout = process_df_in_parallel(tp63, gene_expression_classifier)

sort_gout = gout.sort_values(by='likelihood_ratio', ascending=False)

print sort_gout.head(n=10)
print sort_gout.tail(n=10)

end = time.time()

print end - start
