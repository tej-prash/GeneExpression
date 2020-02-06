from keras.callbacks import LearningRateScheduler, ModelCheckpoint,CSVLogger
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback

base_path="./tej_tests/GeneDataset/method_5/"

class AbstractModel:
    """
    The base class for all Model classes
    """

    def __init__(self, path: str, n_classes: int = 2, activation='relu', task: str = 'classification',
                 bs: int = 64, train_size: float = 0.7, optimizer: str = 'sgd', epochs: int = 100,
                 balance: bool = True):
        """
        Initializes a Model instance.
        :param path: Path to the CSV file
        :param n_classes: Number of classes.
        :param activation: Activation to use
        :param task: Type of task to perform
        :param bs: Batch size
        :param train_size: Training set split size
        :param optimizer: Optimizer for neural network
        :param epochs: Number of epochs
        :param balance: Boolean. If True, balance the dataset before classification
        """

        if not os.path.exists(path):
            raise FileNotFoundError('Path does not exist')
        if n_classes < 2:
            raise ValueError('n_classes must be at least 2.')
        if optimizer == 'sgd':
            self.optimizer = SGD(clipnorm=2.0)
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
        self.balance = balance

        self.lr_history = []
        self.model = None

        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.x_train = self.x_test = self.y_train = self.y_test = None

    def _get_model(self):
        """
        Get a model instance.
        :return: Keras Model instance
        """
        pass

    def _lr_schedule(self, epoch: int, base_lr: int, data=None):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch: int. Epoch number
        :return: learning rate
        """
        if data is None:
            data = self.x_train

        if self.task == 'regression':
            return 0.1

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

    def record_output(self,epoch,logs):
        pass
    
    def constant_LR_decay(self,epoch: int,decay_factor: float):
        """
        Return the learning rate for decay based scheduler
        """
        pass

    def fit(self, finish_fit: bool = True):
        """
        Fit to data.

        The parameter finish_fit is used in cases where you don't want to actually call
        model.fit(), but want to do something else instead. For an example, see
        symnet/image/resnet.py for an example. In most cases, you'll want to leave this
        True. THIS PARAMETER IS ONLY A HACK, NOT A FEATURE.

        :param finish_fit: bool. Set to True unless you know what you're doing.
        :return: None
        """

        if self.x_train is None or self.x_test is None or \
           self.y_train is None or self.y_test is None:
            # Again a hack: these aren't set for image classification.
            if finish_fit:
                raise ValueError('Data is None')

        self.model = self._get_model()

        lr_scheduler = LearningRateScheduler(self._lr_schedule)
        csv_logger=CSVLogger(filename=base_path+'training_adaptive.log',append='True')
        individual_record_callback=LambdaCallback(on_epoch_end=self.record_output)

        # Prepare callbacks for model saving and for learning rate adjustment.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'model.{epoch:03d}.h5'

        # Prepare model model saving directory.
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)
        print("self.optimizer",self.optimizer)
        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

        if finish_fit:
            self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.epochs,
                           batch_size=self.bs, shuffle=True, callbacks=[lr_scheduler, checkpoint,csv_logger,individual_record_callback])

        # Save model 

        # Save model weights
        self.model.save_weights(base_path+"adaptive/trial_1/model_weights.h5")

        # Save model architecture as json
        model_json = self.model.to_json()
        with open(base_path+"adaptive/trial_1/model.json","w") as json_file:
            json_file.write(model_json)

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

        if self.x_test is None or self.y_test is None:
            raise ValueError('Test data is None')

        return self.model.evaluate(self.x_test, self.y_test, batch_size=self.bs)

    def plot_lr(self):
        """
        Plots learning rate history
        :return: None
        """
        plt.style.use('ggplot')
        plt.plot(range(self.epochs), self.lr_history)
