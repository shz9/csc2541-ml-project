import pandas as pd
from sklearn.preprocessing import scale


def read_tp63_dataset(drop_nan=False):
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

    if drop_nan:
        tp63_data = tp63_data.dropna()
    else:
        # Filter out profiles where more than half of
        # the measurements are NaN:
        tp63_data = tp63_data.loc[tp63_data.count(1) > tp63_data.shape[1] / 2, ]

    return tp63_data


def read_yeast_cell_cycle_dataset(time_course="alpha", drop_nan=False):
    """

    Possible values for the time_course parameter:
    - alpha
    - cdc15
    - cdc28
    - elu

    The dataset was retrieved from
    http://genome-www.stanford.edu/cellcycle/data/rawdata/
    Retrieval date: Nov. 14, 2017

    """

    yeast_data = pd.read_csv("yeast_data.txt",
                             header=0,
                             index_col=0,
                             sep="\t")

    # Filter out other time courses / delimiter columns:
    yeast_data = yeast_data.loc[:, [cl for cl in yeast_data.columns
                                    if time_course in cl and time_course != cl]]

    if drop_nan:
        yeast_data = yeast_data.dropna()
    else:
        # Filter out profiles where more than half of
        # the measurements are NaN:
        yeast_data = yeast_data.loc[yeast_data.count(1) > yeast_data.shape[1] / 2, ]

    return yeast_data
