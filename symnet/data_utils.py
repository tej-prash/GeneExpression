# General data utils functions
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.utils import to_categorical
import numpy as np


def rebalance(frame: pd.DataFrame, col: str):
    """
    Rebalance a DataFrame with imbalanced records
    :param frame: DataFrame.
    :param col: The target column
    :return: Balanced DataFrame
    """
    max_size = frame[col].value_counts().max()
    lst = [frame]
    for _, group in frame.groupby(col):
        lst.append(group.sample(int(max_size - len(group)), replace=True))
    frame_new = pd.concat(lst)

    return frame_new


def read_data(path: str, balance: bool = True, train_size: float = 0.7,
):
    """
    Reads a CSV data file, optionally balance it, and split into train/test sets.
    :param path: str. Path to the CSV file
    :param label_column: str. Label column
    :param header: Boolean. True if CSV file has a header
    :param balance: Boolean. True if data should be rebalanced
    :param train_size: float. Percentage of data to be taken as training set
    :param categorize: bool. If True, uses to_categorical to generate one-hot encoded outputs
    :param file_type: str. If binary, a binary npy file is read. If csv, a csv file is read. 
    :return: (X, y) tuple
    """

        
 
    print("Reading a binary file")
    # Read a npy file 
    x_train_path,y_train_path,x_val_path,y_val_path=tuple(path.split(","))
    print(x_train_path,y_train_path,x_val_path,y_val_path)
    
    x_train=np.load(x_train_path)
    y_train=np.load(y_train_path)
    x_test=np.load(x_val_path)
    y_test=np.load(y_val_path)

    return x_train, x_test, y_train, y_test
