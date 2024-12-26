# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:20:35 2021

@author: Nielsen
"""

from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import backend as K

# Criando uma função de ativação
class GeluActivation(layers.Activation):
    def __init__(self, activation, **kwargs):
        super(GeluActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'

    def get_config(self):
        config = super(GeluActivation, self).get_config()
        return config

        
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x,3)) ))

get_custom_objects().update({'gelu' : GeluActivation(gelu)})

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="attention_weight"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="attention_bias"
        )
        self.u = self.add_weight(
            shape=(self.units, 1),
            initializer="random_normal",
            trainable=True,
            name="attention_score"
        )

    def call(self, inputs):
        ui = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        scores = tf.nn.softmax(tf.squeeze(tf.matmul(ui, self.u), axis=-1), axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(scores, -1), axis=1)
        return context

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            "units": self.units
        })
        return config


# Criando a camada de atenção
# class Attention(layers.Layer):
#     def __init__(self, attention_dim=50, **kwargs):
#         self.init = tf.keras.initializers.get('normal')
#         self.attention_dim = attention_dim
        
#         super(Attention, self).__init__(**kwargs)
    
#     def build(self, input_shape):
#         self.W = K.variable(self.init((input_shape[-1],1)))
#         self.b = K.variable(self.init((self.attention_dim, )))
#         self.u = K.variable(self.init((self.attention_dim, 1)))
        
#         super(Attention, self).build(input_shape)
#     def call(self, x):
#         ui = K.tanh(K.bias_add(K.dot(x,self.W), self.b))
#         dot = K.squeeze(K.dot(ui, self.u), -1)
        
#         attention_weights = tf.nn.softmax(dot, axis=1)
#         attention_weights = K.expand_dims(attention_weights)
        
#         context_vector = attention_weights * x
#         context_vector = K.sum(context_vector, axis=1)
        
#         return context_vector
        
    
#     def get_config(self):
#         return {'attention_dim' : self.aattention_dim}

