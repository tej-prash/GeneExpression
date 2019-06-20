"""
Numeric (tabular) data utilities.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize(data):
    """
    Normalize the data
    :param data: array-like. The data, excluding the labels
    :return: Normalized data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)
