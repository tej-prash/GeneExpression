from symnet.data_utils import read_data,normalize,normalize_fit
from keras.models import Model,model_from_json
from symnet import AbstractModel, CustomActivation
from symnet.activations import ARelu,SBAF
from keras.callbacks import LearningRateScheduler, LambdaCallback , TensorBoard
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
from keras.activations import relu
# from mlsquare.losses.keras import quantile_loss
import time

base_path="./tej_tests/GeneDataset/keras_2.3.0/method_19/X2/"

class RegressionModel(AbstractModel):
    """
        RegressionModel: Uses LipschitzLR policy for regression tasks
    """
    def __init__(self, path: str, n_classes: int = 0, activation='relu', task: str = 'regression',
                 bs: int = 64, train_size: float = 0.7, optimizer: str = 'sgd', epochs: int = 100,
                 balance: bool = True,label_column:str=None,header:int=0,f_type='csv',flag_type = "adaptive"):
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
        :param f_type: str. Type of file(binary/csv)
        """

        if not os.path.exists(path):
            if(f_type=='csv'):
                raise FileNotFoundError('Path does not exist')
        if n_classes > 1:
            raise ValueError('n_classes must be 1.')
        if optimizer == 'sgd':
            self.optimizer = SGD()
            self.optimizer_name = 'sgd'
        elif optimizer == 'Adam':
            self.optimizer=Adam()
            self.optimizer_name = 'Adam'
            # self.K_1=[]
            # self.K_2=[]
            # self.beta_1=0.7
            # self.beta_2=0.9
            # self.epsilon=1e-8
        elif optimizer == 'AdaMo':
            self.beta = 0.5
            self.optimizer = SGD(momentum=self.beta)
            self.optimizer_name = 'AdaMo'
            print("Beta",self.beta)
        elif optimizer == 'AdaDelta':
            self.optimizer=Adadelta(learning_rate=1.0)
            self.optimizer_name='AdaDelta'
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
        self.file_type=f_type
        self.balance = balance

        self.lr_history = []
        self.K_z=[]
        self.model = None
        self.lr_time=[] 

        self.loss = 'mean_absolute_error'
        self.metrics = ['mean_absolute_error']

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.label_column = label_column
        self.flag_type=flag_type

        self.x_train, self.x_test, self.y_train, self.y_test = \
            read_data(path, label_column, header, balance=self.balance, train_size=self.train_size,categorize=False,file_type=self.file_type) 
        
        # Logging data header
        # with open(base_path+"output_record","a") as fp:
        #     fp.write("Epoch,")

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
        # self.scaler_x,self.scaler_y=tuple(map(normalize_fit,[self.x_train,self.y_train.reshape(-1,1)]))
        # self.x_train,self.y_train=tuple(map(normalize,[self.x_train,self.y_train.reshape(-1,1)],[self.scaler_x,self.scaler_y]))
        # self.x_test,self.y_test=tuple(map(normalize,[self.x_test,self.y_test.reshape(-1,1)],[self.scaler_x,self.scaler_y]))
        
        print(self.y_train.shape)

    def _get_model(self):
        """
            Initialize a Keras model instance
        """
        ## Single layer network architecture
        # gu=glorot_uniform(seed=54)

        # x = Input(shape=(self.x_train.shape[1],))

        # hidden_out_1=Dense(1000)(x)
        # act_1 = CustomActivation('arelu')(hidden_out_1)
        # dropout_1=Dropout(0.1)(act_1)

        # hidden_out_2=Dense(1000)(dropout_1)
        # act_2 = CustomActivation('arelu')(hidden_out_2)
        # dropout_2=Dropout(0.1)(act_2)

        # y=Dense(self.y_train.shape[1],activation='softsign')(dropout_2)

        x = Input(shape=(self.x_train.shape[1],))

        hidden_out_1 = Dense(3000)(x)
        act_1 = Activation('tanh')(hidden_out_1)
        dropout_1 = Dropout(0.1)(act_1)

        hidden_out_2 = Dense(3000)(dropout_1)
        act_2 = Activation('tanh')(hidden_out_2)
        dropout_2 = Dropout(0.1)(act_2)

        # hidden_out_1=Dense(2000)(x)
        # bn_1 = BatchNormalization()(hidden_out_1)
        # act_1 = Activation(lambda x:relu(x,alpha=0.1,max_value=3.5))(bn_1)
        # dropout_1=Dropout(0.1)(act_1)

        # hidden_out_2=Dense(2000)(dropout_1)
        # bn_2 = BatchNormalization()(hidden_out_2)
        # act_2 = Activation(lambda x:relu(x,alpha=0.1,max_value=3.5))(bn_2)
        # dropout_2=Dropout(0.1)(act_2)

        # hidden_out_3=Dense(1000)(dropout_2)
        # bn_3 = BatchNormalization()(hidden_out_3)
        # act_3 = Activation(lambda x:relu(x,alpha=0.1,max_value=3.5))(bn_3)
        # dropout_3=Dropout(0.1)(act_3)

        y=Dense(self.y_train.shape[1],activation='softsign')(dropout_2)

        self.model = Model(inputs=x,outputs=y)

        plot_model(self.model,to_file=base_path+"model_img.png",show_shapes=True,show_layer_names=True)

        if(self.flag_type!='adaptive'):
            self.model.load_weights(base_path+"constant/trial_0.1/model_weights.h5")

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

    def record_output(self,epoch,logs):
        """
        Log output of every output unit
        :param epoch : int Epoch number
        :param logs : logs collected during epoch
        """
        if(len(self.model.layers)!=2):
                activ_func = K.function([self.model.layers[0].input], [self.model.output])

        # Training data
        end_i = self.x_train.shape[0]
        diff = self.x_train.shape[0]%self.bs

        if(diff==0):
            start_i = end_i - self.bs
        else:
            start_i = end_i - diff

        xb=self.x_train[start_i:end_i]
        y=self.y_train[start_i:end_i]
        if(len(self.model.layers)>2):  
            ##Using the theoretical framework
            # activ=np.linalg.norm(penultimate_activ_func([xb]),axis=0)
            # Kz=np.max(activ)

            ##Using the previous code
            activ=activ_func([xb])
        
        activ_mean_train=np.mean(activ[0],axis=0)
        y_mean_train = np.mean(y,axis=0)

        # Validation data
        end_i = self.x_test.shape[0]
        diff = self.x_test.shape[0]%self.bs

        if(diff==0):
            start_i = end_i - self.bs
        else:
            start_i = end_i - diff

        xb=self.x_test[start_i:end_i]
        y=self.y_test[start_i:end_i]
        if(len(self.model.layers)>2):  
            ##Using the theoretical framework
            # activ=np.linalg.norm(penultimate_activ_func([xb]),axis=0)
            # Kz=np.max(activ)

            ##Using the previous code
            activ=activ_func([xb])
        
        activ_mean_test=np.mean(activ[0],axis=0)
        y_mean_test = np.mean(y,axis=0)
        
        f_name=''
        if(self.flag_type == "adaptive"):
            f_name="adaptive/trial_1/output_record_adaptive"
        else:
            _,lr=self.flag_type.split(";")
            f_name="constant/trial_" + str(lr) +"/output_record_constant"

        with open(base_path+f_name,"a") as fp:
            # Epoch,Predicted_train,y_train,predicted_test,y_test
            fp.write(str(epoch)+",")
            print(activ_mean_train.shape,activ_mean_test.shape,y_mean_train.shape,y_mean_test.shape)
            for i in activ_mean_train:
                fp.write(str(i)+",")
            for i in y_mean_train:
                fp.write(str(i)+",")
            for i in activ_mean_test:
                fp.write(str(i)+",")
            for i in y_mean_test:
                fp.write(str(i)+",")
            fp.write("\n")

    def constant_LR_decay(self,epoch: int,decay_factor: float):
        """
        Return the learning rate for decay based scheduler
        """
        initial_LR = 5 * (1e-4)
        if(epoch==0):
            self.model.save_weights(base_path+"constant/trial_1/model_constant.h5")
            self.lr_history.append(initial_LR)
            return self.lr_history[-1]
        lr = self.lr_history[-1] * decay_factor
        if (lr < 1e-5):
            lr = 1e-5
        self.lr_history.append(lr)
        return lr 


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
                    self.model.save_weights(base_path+"adaptive/trial_3/model_adaptive.h5")
            
            else:
                _,lr=self.flag_type.split(";")
                if(epoch == 0):
                    # self.model.save_weights(base_path+"constant/trial_" + lr +  "/model_constant.h5")
                    print(lr)   
                self.end_lr_time=time.time()
                self.lr_time.append(self.end_lr_time - self.start_lr_time)
                return float(lr)

            # constant LR with decay
            # return self.constant_LR_decay(epoch,0.9)
            # if(epoch==0):
            #     self.model.save_weights(base_path+"constant/trial_1/model_constant.h5")

            # return 10.0

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
            K_max=-1.0
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
                    # print(activ.shape)
                    # print(penultimate_activ_func([xb]))
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
                L=Kz
                if(L>K_max):
                    K_max=L

            # penul_output=z_activ_func([self.x_train[64:128]])
            # penul_output=penultimate_activ_func([self.x_train[:64]])
            # print("Output is",penul_output)
            # print("Label is",self.y_train[:64])
            # print("Z is ",penul_output)
            K_max = K_max/(self.bs * self.y_train.shape[1])
            lr=float(1/K_max)
            # if(epoch == 0):
            #     lr = lr * 1e-3
            lr = lr * 0.01
            self.end_lr_time=time.time()
            self.lr_time.append(self.end_lr_time - self.start_lr_time)
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
                self.model.save_weights(base_path+"constant/trial_1/model_constant.h5")

            return 1e-3

            # if(epoch==0):
            #     self.model.save_weights(path+"/model_constant.h5")

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
            # if(epoch==0):
            #     self.model.save_weights(base_path+"adaptive/trial_1/model_adaptive.h5")
            # constant LR with decay
            print("In AdaMo!")
            return self.constant_LR_decay(epoch,0.9)

            # if(epoch==0):
            #     self.model.save_weights(base_path+"model_constant.h5")

            # return 1e-4

            if(len(self.model.layers)!=2):
                penultimate_activ_func = K.function([self.model.layers[0].input], [self.model.layers[-2].output])

            #Maximum Lipschitz constant
            K_max=-1.0
            for i in range(((len(self.x_train) - 1) // self.bs + 1)):
                start_i=i*self.bs
                end_i=start_i+self.bs
                xb=self.x_train[start_i:end_i]
                # y=self.y_train[start_i:end_i]
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


                L=Kz
                if(L>K_max):
                    K_max=L
            
            K_max = K_max/(self.bs * self.y_train.shape[1])
            K_prime = 0.0
            if(epoch == 0):
                K_prime = K_max 
            else:
                K_prime = (self.beta * self.K_z[-1]) + ((1 - self.beta)*(K_max))

            lr=float(1/K_prime)
            print("Kprime",K_prime)
            print("Learning Rate new:",lr)

            self.lr_history.append(lr)
            self.K_z.append(K_prime)

            return lr

        elif(self.optimizer_name == 'AdaDelta'):
            if(self.flag_type!='adaptive'):
                 _,lr=self.flag_type.split(";")
                 if(epoch == 0):
                     self.model.save_weights(base_path+"constant/trial_" + lr +  "/model_constant.h5")
                     print(lr)   
                 self.end_lr_time=time.time()
                 self.lr_time.append(self.end_lr_time - self.start_lr_time)
                 return float(lr)

    def plot_Kz(self):
        """
        Plots Kz
        :return: None
        """
        with open(base_path+"adaptive/trial_2/K_values","a") as fp:
            fp.write("K_z\n")
            for i in self.K_z:
                fp.write(str(i)+"\n")
        plt.plot(np.arange(len(self.K_z)),self.K_z)
        plt.xlabel("Iteration")
        plt.ylabel("K_z")
        plt.title("K_z over time")
        plt.savefig(base_path+"adaptive/trial_2/K_values.png")

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
