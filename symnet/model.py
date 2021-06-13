from keras.callbacks import LearningRateScheduler, ModelCheckpoint,CSVLogger, TensorBoard, Callback
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback

base_path="./test_results/LipGeneModel/"


class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 1

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            save_dir = os.path.join(base_path+"adaptive/", 'saved_models')
            self.model.save_weights(os.path.join(save_dir,name))
        self.batch += 1

class TimeHistory(Callback):
    def on_train_begin(self,logs={}):
        self.time_history=[]
    def on_epoch_begin(self,epoch,logs={}):
        self.epoch_start_time=time.time()
    def on_epoch_end(self,epoch,logs={}):
        self.time_history.append(time.time() - self.epoch_start_time)

class AbstractModel:
    """
    The base class for all Model classes
    """

    def __init__(self, path: str, bs: int = 64, train_size: float = 0.7, optimizer: str = 'sgd', epochs: int = 100,flag_type = "adaptive"):
        """
        Initializes a Model instance.
        :param bs: Batch size
        :param train_size: Training set split size
        :param optimizer: Optimizer for neural network
        :param epochs: Number of epochs
        """


        if optimizer == 'sgd':
            # self.optimizer = SGD(clipnorm=2.0)
            self.optimizer = SGD()
        else:
            raise NotImplementedError('Only SGD optimizer is implemented!')
        if bs < 1 or not isinstance(bs, int):
            raise ValueError('Improper batch size')
        if train_size < 0 or train_size > 1:
            raise ValueError('Improper train_size argument')
        if epochs < 1:
            raise ValueError('Invalid number of epochs')

        self.bs = bs
        self.train_size = train_size
        self.epochs = epochs

        self.lr_history = []
        self.model = None
        self.lr_time=[]

        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.flag_type=flag_type

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
        symnet/image/resnet.py for an example. 

        :param finish_fit: bool. Set to True unless you know what you're doing.
        :return: None
        """

        if self.x_train is None or self.x_test is None or \
           self.y_train is None or self.y_test is None:
            if finish_fit:
                raise ValueError('Data is None')

        self.model = self._get_model()

        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        if(self.flag_type == "adaptive"):
            csv_logger=CSVLogger(filename=base_path+'adaptive/training_adaptive.log',append='True')
            save_dir = os.path.join(base_path+"adaptive/", 'saved_models')

        else:
            _,lr=self.flag_type.split(";")
            csv_logger=CSVLogger(filename=base_path+'constant/'+  'trial_' + lr + '/training_constant.log',append='True')
            save_dir = os.path.join(base_path+'constant/'+  'trial_' + lr , 'saved_models')

      
        # Prepare callbacks for model saving and for learning rate adjustment.
        model_name = 'model_best.h5'

        # # Prepare model model saving directory.
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='loss',
                                     verbose=1,
                                     save_best_only=True,save_weights_only=True)



        print("self.optimizer",self.optimizer,self.loss,self.metrics)
        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

        if finish_fit:
            time_callback = TimeHistory()
            self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=self.epochs,
                           batch_size=self.bs, shuffle=True, callbacks=[csv_logger,time_callback,checkpoint,lr_scheduler])
            string_to_dump=""
            for i in range(len(time_callback.time_history)):
                string_to_dump += str(i)+","+str(time_callback.time_history[i] + self.lr_time[i])+"\n"
        # Save model 

        # Save model weights
        if(self.flag_type == "adaptive"):
            self.model.save_weights(base_path+"adaptive/model_weights.h5")

            # Save model architecture as json
            model_json = self.model.to_json()
            with open(base_path+"adaptive/model.json","w") as json_file:
                json_file.write(model_json)

            with open(base_path+"adaptive/epoch_times_adaptive.log","a") as fp:
                fp.write(string_to_dump)
        else:
            _,lr=self.flag_type.split(";")

            self.model.save_weights(base_path+"constant/trial_" + lr + "/model_weights.h5")

            # Save model architecture as json
            model_json = self.model.to_json()
            with open(base_path+"constant/trial_"+ lr + "/model.json","w") as json_file:
                json_file.write(model_json)
            
            with open(base_path+"constant/trial_"+ lr + "/epoch_times.log","a") as fp:
                    fp.write(string_to_dump)

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