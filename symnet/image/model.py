from symnet.numeric.data_utils import normalize, read_data
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input, Dropout, Concatenate
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import os


class NumericModel:
    """
    NumericModel: for tabular data sets. Uses a standard ResNet-like architecture with reasonable default settings.
    Uses the LipschitzLR policy: https://arxiv.org/abs/1902.07399
    """
    def __init__(self, path: str, n_classes: int = 2, label_column: str = None, header: bool = True,
                 activation: str = 'relu', task: str = 'classification', bs: int = 64, train_size: float = 0.7,
                 optimizer: str = 'sgd', epochs: int = 10):
        if task == 'regression':
            raise NotImplementedError('Regression is not supported yet!')
        if task != 'classification':
            raise ValueError('Invalid task')
        if not os.path.exists(path):
            raise FileNotFoundError('Path does not exist')
        if n_classes < 2:
            raise ValueError('n_classes must be at least 2.')
        if optimizer == 'sgd':
            self.optimizer = SGD()
        else:
            raise NotImplementedError('Only SGD optimizer is implemented!')
        if bs < 1 or not isinstance(bs, int):
            raise ValueError('Improper batch size')
        if train_size < 0 or train_size > 1:
            raise ValueError('Improper train_size argument')
        if epochs < 1:
            raise ValueError('Invalid number of epochs')

        self.x_train, self.x_test, self.y_train, self.y_test = self._get_data(path, label_column, header)
        self.activation = activation
        self.task = task
        self.bs = bs
        self.train_size = train_size
        self.n_classes = n_classes
        self.epochs = epochs

        self.lr_history = []
        self.model = None

        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

    def _lr_schedule(self, epoch: int):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch: int. Epoch number
        :return: learning rate
        """
        penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

        Kz = 0.
        for i in range((len(self.x_train) - 1) // self.bs + 1):
            start_i = i * self.bs
            end_i = start_i + self.bs
            xb = self.x_train[start_i:end_i]

            activ = np.linalg.norm(penultimate_activ_func([xb]))
            if activ > Kz:
                Kz = activ

        K_ = ((self.n_classes - 1) * Kz) / (self.n_classes * self.bs)
        lr = 1 / K_

        self.lr_history.append(lr)
        return lr

    def _get_data(self, path: str, label_column: str = None, header: bool = True):
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

    def fit(self):
        """
        Fit to data
        :return: None
        """
        self.model = self._get_model()
        lr_scheduler = LearningRateScheduler(self._lr_schedule)
        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.epochs,
                       batch_size=self.bs, callbacks=[lr_scheduler])

    def predict(self, x: np.ndarray):
        """
        Predict on new data
        :param x: array-like
        :return:
        """
        return self.model.predict(x)
