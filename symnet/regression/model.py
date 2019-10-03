from symnet.data_utils import read_data,normalize
from keras.models import Model
from symnet import AbstractModel, CustomActivation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.layers import Dense, Input, Dropout, Concatenate
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.initializers import glorot_uniform

class RegressionModel(AbstractModel):
    """
        RegressionModel: Uses LipschitzLR policy for regression tasks
    """
    def __init__(self, path: str, n_classes: int = 0, activation='relu', task: str = 'regression',
                 bs: int = 64, train_size: float = 0.7, optimizer: str = 'sgd', epochs: int = 100,
                 balance: bool = True,label_column:str=None,header:int=0):
        """
        Initializes a RegressionModel instance.
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
        if n_classes > 1:
            raise ValueError('n_classes must be 1.')
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
        self.balance = balance

        self.lr_history = []
        self.model = None

        self.loss = 'mean_absolute_error'
        self.metrics = ['mean_absolute_error']

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.label_column = label_column
        self.x_train, self.x_test, self.y_train, self.y_test = \
            read_data(path, label_column, header, balance=self.balance, train_size=self.train_size,categorize=False) 

        self.x_train,self.x_test,self.y_train,self.y_test=tuple(map(normalize,[self.x_train,self.x_test,self.y_train.reshape(-1,1),self.y_test.reshape(-1,1)]))

    def _get_model(self):
        """
            Initialize a Keras model instance
        """
        gu=glorot_uniform(seed=54)

        x = Input(shape=(self.x_train.shape[1],))
        z=Dense(7,activation='relu',kernel_initializer=gu)(x)
        y=Dense(1,activation=self.activation,kernel_initializer=gu)(z)

   

        return Model(inputs=x,outputs=y)

    def _lr_schedule(self, epoch: int):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch: int. Epoch number
        :return: learning rate
        """

        # if self.task == 'regression':
        #     # TODO: Implement this with LipschitzLR
        #     #Find n1 and n2
        #     self.model.inputs
        #     self.model.outputs


        if self.x_train is None:
            raise ValueError('x_train is None')

        if(len(self.model.layers)!=2):
            penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

       
        #Maximum Lipschitz constant
        K_max=-1
        for i in range(((len(self.x_train) - 1) // self.bs + 1)):
            start_i=i*self.bs
            end_i=start_i+self.bs
            xb=self.x_train[start_i:end_i]
          
            if(len(self.model.layers)>2):  
                ##Using the theoretical framework
                # activ=np.linalg.norm(penultimate_activ_func([xb]),axis=0)
                # Kz=np.max(activ)

                ##Using the previous code
                activ=np.linalg.norm(penultimate_activ_func([xb]))
                Kz=activ

            else:
                ##Using the theoretical framework
                # activ=np.linalg.norm((xb),axis=0)
                # Kz=np.max(activ)

                ##Using the previous code
                activ=np.linalg.norm([xb])
                Kz=activ


            # print("kz is :",Kz)
            L=Kz/float(self.bs);
            if(L>K_max):
                K_max=L

        lr=float(1/K_max)
        print("Kmax",K_max)
        print("Learning Rate new:",lr)
        self.lr_history.append(lr)
        return lr


        # Kz = 0.
        # for i in range((len(self.x_train) - 1) // self.bs + 1):
        #     start_i = i * self.bs
        #     end_i = start_i + self.bs
        #     xb = self.x_train[start_i:end_i]

        #     activ = np.linalg.norm(penultimate_activ_func([xb]))
        #     if activ > Kz:
        #         Kz = activ

        # K_ = ((self.n_classes - 1) * Kz) / (self.n_classes * self.bs)
        # lr = 1 / K_

        # self.lr_history.append(lr)
        # return lr