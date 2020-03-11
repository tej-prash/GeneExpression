# General data utils functions
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.utils import to_categorical
import numpy as np


def normalize_fit(data):
    """
    Normalize the data
    :param data: array-like. The data, excluding the labels
    :return: Normalized data
    """
    #Feature Scaling using normalisation-Activation: Linear
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

def normalize(data,scaler):
    """
    Normalize the data
    :param data: array-like. The data, excluding the labels
    :return: Normalized data
    """
    #Feature Scaling using normalisation-Activation: Linear
    return scaler.transform(data)


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


def read_data(path: str, label_column: str = None, header: int = 0, balance: bool = True, train_size: float = 0.7,
              categorize=True,file_type='csv'):
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
    if file_type=='csv' and (path is None or not os.path.exists(path)):
        print('WARNING: Path does not exist, or is None.')
        return [[]], []
        
    if(file_type=='b'):
        print("Reading a binary file")
        # Read a npy file 
        x_train_path,y_train_path,x_val_path,y_val_path=tuple(path.split(","))
        print(x_train_path,y_train_path,x_val_path,y_val_path)
        
        x_train=np.load(x_train_path)
        y_train=np.load(y_train_path)
        x_test=np.load(x_val_path)
        y_test=np.load(y_val_path)

    elif(file_type=='csv'):
        df = pd.read_csv(path, header=header)
        if len(df.columns) == 0:
            print('WARNING: File has no columns')
            return [[]], []

        if label_column is None:
            label_column = df.columns[-1]

        # Normalizing before splitting
        # Feature scaling technique-MinMax Scaler
        # scaler=MinMaxScaler()
        # df_X=df.loc[:,df.columns!=label_column]
        # print("df_X",df_X.columns)
        # scaled_values=scaler.fit_transform(df_X.values)
        # scaled_df=pd.DataFrame(data=scaled_values,columns=df_X.columns)
        # scaled_df[label_column]=df[label_column]
        # print("scaled_df",scaled_df.columns)
        # print(scaled_df[label_column])
        # scaled_df.loc[:,label_column]=scaled_df.loc[:,label_column]/float(1e5)
        # print(scaled_df[label_column])

        # Feature scaling technique-Sum of features is 1
        # print(df.columns)
        # print("Label:",label_column)
        # for column in df.columns:
        #     if(column!=label_column):
        #         print(column)
        #         df.loc[:,column]=df.loc[:,column]/float(sum(df.loc[:,column]))
        #     else:
        #         df.loc[:,column]=df.loc[:,column]/float(1e5)
        # print(df.loc[:,label_column])

        # Feature scaling-Standard Scaling
        # scaler=StandardScaler()
        # scaled_values=scaler.fit_transform(df.values)
        # scaled_df=pd.DataFrame(data=scaled_values,columns=df.columns)

        # Normalizing y by dividing by a constant
        # df.loc[:,label_column]=df.loc[:,label_column]/float(1e5)

        train_df, test_df = train_test_split(df, train_size=train_size,random_state=42,shuffle=True)

        if balance:
            print("Balancing dataset")
            train_df = rebalance(train_df, label_column)

        y = train_df[label_column]
        y_test = test_df[label_column]
        train_df.drop(label_column, axis=1, inplace=True)
        test_df.drop(label_column, axis=1, inplace=True)

        if categorize:
            y_train = to_categorical(np.array(y))
            y_test = to_categorical(np.array(y_test))
        else:
            y_train = np.array(y)
            y_test = np.array(y_test)

        x_train = np.array(train_df)
        x_test = np.array(test_df)

    return x_train, x_test, y_train, y_test
