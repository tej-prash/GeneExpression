from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import os
import matplotlib.pyplot as plt


class AbstractModel:
    """
    The base class for all Model classes
    """

    def __init__(self, path: str, n_classes: int = 2,
                 activation: str = 'relu', task: str = 'classification', bs: int = 64, train_size: float = 0.7,
                 optimizer: str = 'sgd', epochs: int = 100):
        """
        Initializes a Model instance.
        :param path: Path to the CSV file
        :param n_classes: Number of classes.
        :param label_column: The column with the labels
        :param header: int. Row making the column names
        :param activation: Activation to use
        :param task: Type of task to perform
        :param bs: Batch size
        :param train_size: Training set split size
        :param optimizer: Optimizer for neural network
        :param epochs: Number of epochs
        """
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

    def _get_data(self, path: str, label_column: str = None, header: int = 0):
        """
        Fetch and pre-process data
        :param path: str. Path to CSV
        :param label_column: str. Label column in the dataset
        :param header: Boolean. True if CSV contains a header row
        :return: (X, y)
        """
        pass

    def _get_model(self):
        """
        Get a model instance.
        :return: Keras Model instance
        """
        pass

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
        :return: predictions:  array-like
        """
        return self.model.predict(x)

    def score(self):
        """
        Returns model performance on test set
        :return:
        """
        return self.model.evaluate(self.x_test, self.y_test, batch_size=self.bs)

    def plot_lr(self):
        """
        Plots learning rate history
        :return: None
        """
        plt.style.use('ggplot')
        plt.plot(range(self.epochs), self.lr_history)
