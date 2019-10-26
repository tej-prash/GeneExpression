from keras.layers import Activation, Layer
from keras import backend as K
import tensorflow as tf
import numpy as np 
import random 


# #Defining gradient as an operation
# def relu_grad(op,grad):
#     x=op.inputs[0]
#     k=op.inputs[1]
#     n=op.inputs[2]
#     n_gr=tf_arelu_derivative(x,k,n)
#     return grad * n_gr

# #Register gradient as an operation
# def py_func(func,inp,Tout,stateful=True,name=None,grad=None):
#     rnd_name='FunGrad'+str(random.randint(0,1E2))
#     tf.RegisterGradient(rnd_name)(grad)
#     g=tf.get_default_graph()
#     with g.gradient_override_map({"PyFunc":rnd_name}):
#         return tf.py_function(func,inp,Tout,name)

# def arelu_derivative(x,k,n):
#     if(x>0):
#         return k*n*pow(x,n-1)
#     elif(x<0):
#         return 0.01
#     else:
#         return 0

# def arelu(x,k,n):
#     if(x>0):
#         return k * ( pow(x,n) )
#     elif(x<0):
#         return 0.01 * x
#     else:
#         return 0


# np_arelu=np.vectorize(arelu)
# np_arelu_derivative=np.vectorize(arelu_derivative)

# np_arelu_32=lambda x,k,n:np_arelu(x,k,n).astype(tf.float32)
# np_arelu_derivative_32=lambda x,k,n:np_arelu_derivative(x,k,n).astype(tf.float32)

# #Defining operation as a tensorflow node
# def tf_arelu(x,k,n,name=None):
#     with tf.name_scope("arelu",[x,k,n]) as name:
#         y=py_func(np_arelu_32,[x,k,n],[tf.float32],name=name,grad=relu_grad)
#         y[0].set_shape(x.get_shape())
#         return y[0]

# #Defining operation as a tensorflow node
# def tf_arelu_derivative(x,k,n,name=None):
#     with tf.name_scope("arelu_derivative",[x,k,n]) as name:
#         y=tf.py_function(np_arelu_derivative_32,[x,k,n],[tf.float32],name=name)
#         return y[0]

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
            self.activation = ARelu()
        super(CustomActivation, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return self.activation(x)
