"""
Numeric (tabular) data utilities.
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def normalize(data):
    """
    Normalize the data
    :param data: array-like. The data, excluding the labels
    :return: Normalized data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def read_data(path: str, label_column=None, header=True):
    """
    Reads a CSV data file
    :param path: str. Path to the CSV file
    :param label_column: str. Label column
    :param header: Boolean. True if CSV file has a header
    :return: (X, y) tuple
    """
    if path is None or not os.path.exists(path):
        return [[]], []

    df = pd.read_csv(path, header=header)
    if len(df.columns) == 0:
        return [[]], []

    if label_column is None:
        label_column = df.columns[-1]

    x_cols = list(df.columns)
    x_cols.remove(label_column)

    return df[x_cols], df[label_column]
