from symnet.data_utils import read_data,normalize,normalize_fit
from keras.models import Model,model_from_json
from symnet import AbstractModel, CustomActivation
from symnet.activations import ARelu,SBAF
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
from keras.constraints import NonNeg

path="./tests/EnergyEfficiency/suraj/sgd/trial_5"

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
            self.optimizer_name = 'sgd'
        elif optimizer == 'Adam':
            self.optimizer=Adam()
            self.optimizer_name = 'Adam'
            self.K_1=[]
            self.K_2=[]
            self.beta_1=0.7
            self.beta_2=0.9
            self.epsilon=1e-8
        elif optimizer == 'AdaMo':
            self.beta = 0.5
            self.optimizer = SGD(momentum=self.beta)
            self.optimizer_name = 'AdaMo'
            print("Beta",self.beta)
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
        
        #Feature scaling only on X
        # self.scaler_x=tuple(map(normalize_fit,[self.x_train]))[0]
        # self.x_train=tuple(map(normalize,[self.x_train],[self.scaler_x]))[0]

        # self.scaler_x=tuple(map(normalize_fit,[self.x_test]))[0]
        # self.x_test=tuple(map(normalize,[self.x_test],[self.scaler_x]))[0]

        #Plot distributions of y_train and y_test
        # sns.distplot(self.y_train,hist=False,label="Training dataset")
        # sns.distplot(self.y_test,hist=False,label="Testing dataset")
        # plt.legend()
        # plt.savefig('./tests/method_14/dist_y_train_y_test.png')

        #Feature scaling on X and Y
        self.scaler_x,self.scaler_y=tuple(map(normalize_fit,[self.x_train,self.y_train.reshape(-1,1)]))
        self.x_train,self.y_train=tuple(map(normalize,[self.x_train,self.y_train.reshape(-1,1)],[self.scaler_x,self.scaler_y]))
        self.x_test,self.y_test=tuple(map(normalize,[self.x_test,self.y_test.reshape(-1,1)],[self.scaler_x,self.scaler_y]))
        
        print(self.x_train.shape)

    def _get_model(self):
        """
            Initialize a Keras model instance
        """
        ## Single layer network architecture
        # gu=glorot_uniform(seed=54)

        x = Input(shape=(self.x_train.shape[1],))
        hidden_out_1=Dense(20,activation=self.activation)(x)
        hidden_out_2=Dense(15,activation=self.activation)(hidden_out_1)
        y=Dense(1,activation='softsign')(hidden_out_2)

        self.model = Model(inputs=x,outputs=y)

        plot_model(self.model,to_file="./tej_tests/CaliforniaHousing/method_26/model_img.png",show_shapes=True,show_layer_names=True)

        self.model.load_weights('./tej_tests/CaliforniaHousing/method_26/random_state_42/model_constant.h5') 

        return self.model

        ## RES-Net like architecture

        # inp = Input(shape=(self.x_train.shape[1],))

        # bn1 = BatchNormalization(name='first_bn')(inp)
        # dense = Dense(5, name='dense1')(bn1)
        # act = CustomActivation(self.activation)(dense)
        # drop1 = Dropout(0.2, name='dropout1')(act)

        # bn = BatchNormalization(name='bn1')(drop1)
        # #bn=drop1
        # dense = Dense(5, name='dense2')(bn)
        # act = CustomActivation(self.activation)(dense)
        # drop2 = Dropout(0.2)(act)

        # interm = Concatenate()([drop1, drop2])

        # bn = BatchNormalization(name='bn2')(interm)
        # #bn=interm
        # dense = Dense(5, name='dense3')(bn)
        # act = CustomActivation(self.activation)(dense)
        # drop = Dropout(0.2)(act)

        # interm = Concatenate()([drop, drop2])
        # #bn=interm
        # bn = BatchNormalization()(interm)

        # if self.task == 'classification':
        #     out = Dense(3, activation='softmax', name='dense4')(bn)
        # else:
        #     out = Dense(1,name='dense4',activation='softsign')(bn)

        # self.model = Model(inputs=inp, outputs=out)

        # plot_model(self.model,to_file="./tej_tests/CaliforniaHousing/model_img.png",show_shapes=True,show_layer_names=True)

        # self.model.load_weights('./tej_tests/CaliforniaHousing/method_12/random_state_42/model_constant.h5') 

        # return self.model

    def _lr_schedule(self, epoch: int):
        """
        Get the learning rate for a given epoch. Note that this uses the LipschitzLR policy, so the epoch
        number doesn't actually matter.
        :param epoch: int. Epoch number
        :return: learning rate
        """

        if self.x_train is None:
            raise ValueError('x_train is None')

        if(self.optimizer_name == 'sgd'):
            #Verify model weights
            if(epoch==0):
                self.model.save_weights("./tej_tests/CaliforniaHousing/method_26/random_state_42/model_adaptive.h5")


            # if(epoch==0):
            #     self.model.save_weights("./tej_tests/CaliforniaHousing/method_26/random_state_42/model_constant.h5")

            # return 0.1

            if(len(self.model.layers)!=2):
                penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

            # activ_func=K.function([self.model.layers[0].input],[self.model.output])
            # z_activ_func=K.function([self.model.layers[0].input],[self.model.layers[-2].output])

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

            # penul_output=z_activ_func([self.x_train[64:128]])
            # output=activ_func([self.x_train[:64]])
            # print("Output is",output)
            # print("Label is",self.y_train[:64])
            # print("Z is ",penul_output)

            lr=float(1/K_max)
            print("Kmax",K_max)
            print("Learning Rate new:",lr)

            self.lr_history.append(lr)
            self.K_z.append(K_max)

            #plt.plot(np.arange(len(gradient_list)),gradient_list)
            #plt.show()
            return lr
            #return lr
        elif(self.optimizer_name == 'Adam'):
            # print("----------------------------Adam--------------------------------")
            # Approximating max||delta(L)^2|| to be (max||delta(L)||)^2

            #Verify model weights
            if(epoch==0):
                self.model.save_weights("./tej_tests/CaliforniaHousing/method_19/model_adaptive.h5")


            # if(epoch==0):
            #     self.model.save_weights("./tej_tests/CaliforniaHousing/method_19/model_constant.h5")

            # return 0.001

            if(len(self.model.layers)!=2):
                penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

            #Maximum Lipschitz constant
            K_max=-1.0
            K_1=-1.0
            K_2=-1.0
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

                L=Kz/float(self.bs)
                if(L>K_max):
                    K_max=L
            if(self.K_1==[]):
                # Use norm of the gradient and square of the norm of the gradient as estimates for the first K1 and K2
                self.K_1.append(K_max)
                self.K_2.append(K_max*K_max)
                K_1=K_max
                K_2=K_max * K_max

                return 0.001
            else:
                K_1=(self.beta_1*self.K_1[-1])+((1-self.beta_1)*(K_max))
                K_2=(self.beta_2*self.K_2[-1])+((1-self.beta_2)*(K_max*K_max))
                self.K_1.append(K_1)
                self.K_2.append(K_2)
            
            K_eff=(float(K_1))/((K_2**(0.5))+self.epsilon)
            lr=1/K_eff
            print("Keff",K_eff)
            print("K1",K_1)
            print("K2",K_2)
            print("Learning Rate new:",lr)

            self.lr_history.append(lr)
            self.K_z.append(K_eff)

            return lr

        elif(self.optimizer_name == 'AdaMo'):
            #Verify model weights
            if(epoch==0):
                self.model.save_weights("./tej_tests/CaliforniaHousing/method_30/random_state_42/model_adaptive.h5")


            # if(epoch==0):
            #     self.model.save_weights("./tej_tests/CaliforniaHousing/method_30/random_state_42/model_constant.h5")

            # return 0.1

            if(len(self.model.layers)!=2):
                penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

            #Maximum Lipschitz constant
            K_max=-1.0
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


                L=Kz/float(self.bs)
                if(L>K_max):
                    K_max=L

            K_prime = 0.0
            if(epoch == 0):
                K_prime = K_max 
            else:
                K_prime = (self.beta * self.K_z[-1]) + (1 - self.beta)*(K_max)

            lr=float(1/K_prime)
            print("Kprime",K_prime)
            print("Learning Rate new:",lr)

            self.lr_history.append(lr)
            self.K_z.append(K_prime)

            return lr


    def plot_Kz(self):
        """
        Plots Kz
        :return: None
        """
        with open("./tej_tests/CaliforniaHousing/method_26/random_state_42/K_values","a") as fp:
            fp.write("K_z\n")
            for i in self.K_z:
                fp.write(str(i)+"\n")
        plt.plot(np.arange(len(self.K_z)),self.K_z)
        plt.xlabel("Iteration")
        plt.ylabel("K_z")
        plt.title("K_z over time")
        plt.savefig("./tej_tests/CaliforniaHousing/method_26/random_state_42/K_values.png")

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
