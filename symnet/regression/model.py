from symnet.data_utils import read_data
from keras.models import Model,model_from_json
from symnet import AbstractModel
from keras.optimizers import SGD,Adam,Adadelta
from keras.layers import Dense, Input, Dropout, Concatenate,BatchNormalization,LeakyReLU
import os
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
from keras.initializers import glorot_uniform
from keras.losses import mean_absolute_error
from keras.utils.vis_utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
from keras.constraints import NonNeg
from keras import regularizers
from keras.layers import Activation
import time

base_path="./test_results/LipGeneModel/"

class LipGeneModel(AbstractModel):
    """
        LipGeneModel: Uses LipschitzLR policy for regression tasks
    """
    def __init__(self, path: str, task: str = 'regression',
                 bs: int = 64, train_size: float = 0.7, optimizer: str = 'sgd', epochs: int = 100, flag_type = "adaptive"):
        """
        Initializes a LipGeneModel instance.
        :param path: Path to the binary file
        :param bs: Batch size
        :param train_size: Training set split size
        :param optimizer: Optimizer for neural network
        :param epochs: Number of epochs
        """

        if optimizer == 'sgd':
            self.optimizer = SGD()
            self.optimizer_name = 'sgd'
        elif optimizer == 'Adam':
            self.optimizer=Adam()
            self.optimizer_name = 'Adam'
        elif optimizer == 'AdaMo':
            self.beta = 0.5
            self.optimizer = SGD(momentum=self.beta)
            self.optimizer_name = 'AdaMo'
            print("Beta",self.beta)
        elif optimizer == 'AdaDelta':
            self.optimizer=Adadelta(learning_rate=1.0)
            self.optimizer_name='AdaDelta'
        else:
            raise NotImplementedError('{0} is not implemented!'.format(optimizer))

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
        self.K_z=[]
        self.model = None
        self.lr_time=[] 

        self.loss = 'mean_absolute_error'
        self.metrics = ['mean_absolute_error']

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.flag_type=flag_type

        self.x_train, self.x_test, self.y_train, self.y_test = \
            read_data(path, train_size=self.train_size) 
        
        
        print(self.y_train.shape)

    def _get_model(self):
        """
            Initialize a Keras model instance
        """

        ## Implement your architecture
        x = Input(shape=(self.x_train.shape[1],))

        hidden_out_1 = Dense(3000)(x)
        act_1 = Activation('tanh')(hidden_out_1)
        dropout_1 = Dropout(0.1)(act_1)

        hidden_out_2 = Dense(3000)(dropout_1)
        act_2 = Activation('tanh')(hidden_out_2)
        dropout_2 = Dropout(0.1)(act_2)

        y=Dense(self.y_train.shape[1],activation='softsign')(dropout_2)

        self.model = Model(inputs=x,outputs=y)

        plot_model(self.model,to_file=base_path+"model_img.png",show_shapes=True,show_layer_names=True)

        ## Uncomment this when you want to run adaptive and constant models on the same set of initial weights
        # if(self.flag_type!='adaptive'):
        #     self.model.load_weights(base_path+"adaptive/model_adaptive.h5")

        return self.model

    
    def _lr_schedule(self, epoch: int):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch : int. Epoch number
        :return: learning rate
        """

        if self.x_train is None:
            raise ValueError('x_train is None')
        
        self.start_lr_time=time.time()

        if(self.optimizer_name == 'sgd'):
            #Verify model weights
            if(self.flag_type == "adaptive"):
                if(epoch==0):
                    self.model.save_weights(base_path+"adaptive/model_adaptive.h5")
            
            else:
                _,lr=self.flag_type.split(";")
                if(epoch == 0):
                    self.model.save_weights(base_path+"constant/model_constant.h5")
                    print(lr)   
                self.end_lr_time=time.time()
                self.lr_time.append(self.end_lr_time - self.start_lr_time)
                return float(lr)

            if(len(self.model.layers)!=2):
                penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

            #Maximum Lipschitz constant
            K_max=-1.0
            for i in range(((len(self.x_train) - 1) // self.bs + 1)):
                start_i=i*self.bs
                end_i=start_i+self.bs
                xb=self.x_train[start_i:end_i]
                if(len(self.model.layers)>2):  
                    activ=np.linalg.norm(penultimate_activ_func([xb]))
                    Kz=activ

                else:
                    activ=np.linalg.norm([xb])
                    Kz=activ

                L=Kz
                if(L>K_max):
                    K_max=L

            K_max = K_max/(self.bs * self.y_train.shape[1])
            lr=float(1/K_max)

            ## scaling adaptive learning rate for practical purposes
            lr = lr * 0.01
            self.end_lr_time=time.time()
            self.lr_time.append(self.end_lr_time - self.start_lr_time)
            print("Kmax",K_max)
            print("Learning Rate new:",lr)

            self.lr_history.append(lr)
            self.K_z.append(K_max)

            return lr