from keras.layers import Activation, Layer
from keras import backend as K
import tensorflow as tf
import numpy as np 
import random 


class SBAF(Layer):
    def __init__(self, k: float = 0.91, alpha: float = 0.5, **kwargs):
        self.k = k
        self.alpha = alpha
        super(SBAF, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return 1. / (1+(self.k * K.pow(x, self.alpha) * K.pow(1 - x, 1 - self.alpha)))


class ARelu(Layer):
    def __init__(self, k: float = 0.54, n: float = 1.30, **kwargs):
        self.k = k
        self.n = n
        super(ARelu, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.k * K.pow(x,self.n)


class CustomActivation(Layer):
    """
    A wrapper around Keras activations, that allows us to also use custom activation
    functions.
    """

    def __init__(self, name: str = 'relu', **kwargs):
        if name in ('relu', 'elu', 'selu', 'sigmoid', 'softmax', 'linear', 'softplus'):
            self.activation = Activation(name)
        elif name == 'sbaf':
            self.activation = SBAF()
        elif name == 'arelu':
            self.activation = ARelu(k=0.6,n=1.01)
        super(CustomActivation, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.activation(x)
