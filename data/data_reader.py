"""
Author: Shadi Zabad
Date: November 2017
"""

import pandas as pd
import os


def read_mauna_loa_co2_data():
    co2_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                           'mauna-loa-atmospheric-co2.csv'),
                              header=None)
    co2_dataset.columns = ['CO2Concentration', 'Time']

    return co2_dataset


def read_lake_erie_data():
    erie_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                            'monthly-lake-erie-levels-1921-19.csv'),
                               header=0, skip_footer=3)
    erie_dataset.columns = ['Month', 'Level']

    return erie_dataset


def read_airline_data():
    airline_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                               'airline_data.csv'), header=None)
    airline_dataset.columns = ['Time', 'Passengers']

    return airline_dataset


def read_solar_irradiation_data():
    solar_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                             'solar_irradiance.txt'),
                                sep='     ', header=None, skiprows=7)
    solar_dataset.columns = ['Year', 'TSI']

    return solar_dataset
