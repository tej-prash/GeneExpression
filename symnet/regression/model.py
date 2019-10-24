from symnet.data_utils import read_data,normalize,normalize_fit
from keras.models import Model,model_from_json
from symnet import AbstractModel, CustomActivation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD,Adam
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
        self.K_z=[]
        self.model = None

        self.loss = 'mean_absolute_error'
        self.metrics = ['mean_absolute_error']

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.label_column = label_column

        self.x_train, self.x_test, self.y_train, self.y_test = \
            read_data(path, label_column, header, balance=self.balance, train_size=self.train_size,categorize=False) 

        self.scaler_x,self.scaler_y=tuple(map(normalize_fit,[self.x_train,self.y_train.reshape(-1,1)]))

        #Plot distributions of y_train and y_test
        # sns.distplot(self.y_train,hist=False,label="Training dataset")
        # sns.distplot(self.y_test,hist=False,label="Testing dataset")
        # plt.legend()
        # plt.savefig('./tests/method_14/dist_y_train_y_test.png')

        self.x_train,self.y_train=tuple(map(normalize,[self.x_train,self.y_train.reshape(-1,1)],[self.scaler_x,self.scaler_y]))
        # self.scaler_x,self.scaler_y=tuple(map(normalize_fit,[self.x_test,self.y_test.reshape(-1,1)]))
        
        self.x_test,self.y_test=tuple(map(normalize,[self.x_test,self.y_test.reshape(-1,1)],[self.scaler_x,self.scaler_y]))

        print(self.x_train.shape)

    def _get_model(self):
        """
            Initialize a Keras model instance
        """
        ## Single layer network architecture
        # gu=glorot_uniform(seed=54)

        # x = Input(shape=(self.x_train.shape[1],))
        # z=Dense(7,activation='relu',kernel_initializer=gu)(x)
        # y=Dense(1,activation=self.activation,kernel_initializer=gu)(z)

        # return Model(inputs=x,outputs=y)

        ## RES-Net like architecture

        #Get saved model
        # fp=open("./tests/BostonHousing/method_8/model_constant.json","r")
        # loaded_model=fp.read()
        # fp.close()
        # self.model=model_from_json(loaded_model)


        # return self.model

        inp = Input(shape=(self.x_train.shape[1],))

        bn1 = BatchNormalization(name='first_bn')(inp)
        dense = Dense(5, name='dense1')(bn1)
        act = CustomActivation(self.activation)(dense)
        drop1 = Dropout(0.2, name='dropout1')(act)

        bn = BatchNormalization(name='bn1')(drop1)
        #bn=drop1
        dense = Dense(5, name='dense2')(bn)
        act = CustomActivation(self.activation)(dense)
        drop2 = Dropout(0.2)(act)

        interm = Concatenate()([drop1, drop2])

        bn = BatchNormalization(name='bn2')(interm)
        #bn=interm
        dense = Dense(5, name='dense3')(bn)
        act = CustomActivation(self.activation)(dense)
        drop = Dropout(0.2)(act)

        interm = Concatenate()([drop, drop2])
        #bn=interm
        bn = BatchNormalization()(interm)

        if self.task == 'classification':
            out = Dense(3, activation='softmax', name='dense4')(bn)
        else:
            out = Dense(1,activation='softsign' ,name='dense4')(bn)

        self.model = Model(inputs=inp, outputs=out)

        plot_model(self.model,to_file="./tests/BostonHousing/model_img.png",show_shapes=True,show_layer_names=True)

        self.model.load_weights('./tests/BostonHousing/method_10/model_constant.h5') 

        return self.model

    def _lr_schedule(self, epoch: int):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch: int. Epoch number
        :return: learning rate
        """

        if self.x_train is None:
            raise ValueError('x_train is None')

        #Verify model weights
        if(epoch==0):
            self.model.save_weights("./tests/BostonHousing/method_10/model_adaptive.h5")


        # if(epoch==0):
        #     self.model.save_weights("./tests/BostonHousing/method_10/model_constant.h5")

        # return 0.1

        if(len(self.model.layers)!=2):
            penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

        # activ_func=K.function([self.model.layers[0].input],[self.model.output])
        
        #Calculate gradient
        #grads=K.gradients(self.model.total_loss,self.model.trainable_weights)
        #print("grads",grads)
        #inputs=self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights
        #print("inputs",inputs)
        #grads_func=K.function(inputs,grads)

        #Maximum Lipschitz constant
        K_max=-1
        for i in range(((len(self.x_train) - 1) // self.bs + 1)):
            start_i=i*self.bs
            end_i=start_i+self.bs
            xb=self.x_train[start_i:end_i]
            y=self.y_train[start_i:end_i]
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


            # weight_matrices=[]
            # for layer in self.model.layers:
            #     weight_matrices.append(layer.get_weights())
            # weight_matrices=np.array(weight_matrices)
            # print("hiii")
            # print(weight_matrices.shape)
            # evaluated_grads=grads_func([xb,y,weight_matrices])
            # print(evaluated_grads)    
            #print("kz is :",Kz)
            L=Kz/float(self.bs)
            if(L>K_max):
                K_max=L

        lr=float(1/K_max)
        print("Kmax",K_max)
        print("Learning Rate new:",lr)
        self.lr_history.append(lr)
        self.K_z.append(K_max)

        #plt.plot(np.arange(len(gradient_list)),gradient_list)
        #plt.show()
        return lr
        #return lr
    def plot_Kz(self):
        """
        Plots Kz
        :return: None
        """
        with open("./tests/BostonHousing/method_10/K_values","a") as fp:
            fp.write("K_z\n")
            for i in self.K_z:
                fp.write(str(i)+"\n")
        plt.plot(np.arange(len(self.K_z)),self.K_z)
        plt.xlabel("Iteration")
        plt.ylabel("K_z")
        plt.title("K_z over time")
        plt.savefig("./tests/BostonHousing/method_10/K_values.png")
    # def calculate_loss(self,x:np.ndarray,y:np.ndarray):
    #     """
    #     Predict on new data
    #     :param x: featured-scaled,array-like
    #     :param y: featured-scaled,array-like
    #     :return: predictions:  array-like
    #     """
    #     #reverse transform on y
    #     y_transformed=self.scaler_y_train.inverse_transform(y)
    #     y_pred=self.model.predict(x)
    #     y_pred_transformed=self.scaler_y_train.inverse_transform(y_pred)
    #     total_loss=[abs(y_pred_transformed[i][0]-y_transformed[i][0]) for i in range(len(y_pred))]
    #     return sum(total_loss)/len(total_loss)
