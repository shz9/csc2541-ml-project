import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


def read_tp63_dataset():
    """

    TRP63 Dataset from Della Gatta et al. 2008.
    GEOdataset:GSE10562

    The dataset was transformed by Kalaitzis et al. so that
    it has 14 columns: The Affy Probe ID, and 13 columns
    describing the normalized expression measurements over the
    period of 4 hours. Measurements were taken every 20 minutes.

    """

    tp63_data = pd.read_csv("tp63_exprs_data.csv",
                            header=0,
                            index_col=0)

    tp63_data = pd.DataFrame(scale(tp63_data, axis=1, with_std=False),
                             index=tp63_data.index,
                             columns=tp63_data.columns)

    # Filter out profiles where more than half of
    # the measurements are NaN:

    tp63_data = tp63_data.loc[tp63_data.count(1) > tp63_data.shape[1] / 2, ]

    return tp63_data
