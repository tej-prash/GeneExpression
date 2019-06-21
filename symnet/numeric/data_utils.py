"""
Numeric (tabular) data utilities.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def normalize(data):
    """
    Normalize the data
    :param data: array-like. The data, excluding the labels
    :return: Normalized data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def rebalance(frame: pd.DataFrame, col: str):
    """
    Rebalance a DataFrame with imbalanced records
    :param frame: DataFrame.
    :param col: The target column
    :return: Balanced DataFrame
    """
    max_size = frame[col].value_counts().max()
    lst = [frame]
    for class_index, group in frame.groupby(col):
        lst.append(group.sample(int(max_size - len(group)), replace=True))
    frame_new = pd.concat(lst)

    return frame_new


def read_data(path: str, label_column: str = None, header: int = 0, balance: bool = True, train_size: float = 0.7):
    """
    Reads a CSV data file
    :param path: str. Path to the CSV file
    :param label_column: str. Label column
    :param header: Boolean. True if CSV file has a header
    :param balance: Boolean. True if data should be rebalanced
    :param train_size: float. Percentage of data to be taken as training set
    :return: (X, y) tuple
    """
    if path is None or not os.path.exists(path):
        return [[]], []

    df = pd.read_csv(path, header=header)
    if len(df.columns) == 0:
        return [[]], []

    if label_column is None:
        label_column = df.columns[-1]

    train_df, test_df = train_test_split(df, train_size=train_size)

    if balance:
        train_df = rebalance(train_df, label_column)

    y = train_df[label_column]
    y_test = test_df[label_column]
    train_df.drop(label_column, axis=1, inplace=True)
    test_df.drop(label_column, axis=1, inplace=True)

    y_train = to_categorical(np.array(y))
    y_test = to_categorical(np.array(y_test))

    x_train = np.array(train_df)
    x_test = np.array(test_df)

    return x_train, x_test, y_train, y_test
