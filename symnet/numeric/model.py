from symnet.numeric.data_utils import normalize, read_data
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input, Dropout, Concatenate
import numpy as np
from symnet import AbstractModel
from keras.utils import to_categorical


class NumericModel(AbstractModel):
    """
    NumericModel: for tabular data sets. Uses a standard ResNet-like architecture with reasonable default settings.
    Uses the LipschitzLR policy: https://arxiv.org/abs/1902.07399
    """

    def __init__(self, path: str, label_column: str = None, header: int = 0, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.x_train, self.x_test, self.y_train, self.y_test = self._get_data(path, label_column, header)

    def _get_data(self, path: str, label_column: str = None, header: int = 0):
        """
        Fetch and pre-process data
        :param path: str. Path to CSV
        :param label_column: str. Label column in the dataset
        :param header: Boolean. True if CSV contains a header row
        :return: (X, y)
        """
        x, y = read_data(path, label_column, header)
        x = np.array(x)
        y = np.array(y)
        y = to_categorical(y)
        x = normalize(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)
        return x_train, x_test, y_train, y_test

    def _get_model(self):
        """
        Get a model instance.
        :return: Keras Model instance
        """
        inp = Input(shape=(self.x_train.shape[1],))

        bn1 = BatchNormalization(name='first_bn')(inp)
        relu = Dense(5, activation=self.activation, name='dense1')(bn1)
        drop1 = Dropout(0.2, name='dropout1')(relu)

        bn = BatchNormalization(name='bn1')(drop1)
        relu = Dense(5, activation=self.activation, name='dense2')(bn)
        drop2 = Dropout(0.2)(relu)

        interm = Concatenate()([drop1, drop2])

        bn = BatchNormalization(name='bn2')(interm)
        relu = Dense(5, activation=self.activation, name='dense3')(bn)
        drop = Dropout(0.2)(relu)

        interm = Concatenate()([drop, drop2])

        bn = BatchNormalization()(interm)

        if self.task == 'classification':
            out = Dense(3, activation='softmax', name='dense4')(bn)
        else:
            out = Dense(3, activation=self.activation, name='dense4')(bn)

        return Model(inputs=inp, outputs=out)
