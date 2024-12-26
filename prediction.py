# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:20:51 2021

@author: Nielsen
"""

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import CustomObjectScope
from layers import GeluActivation, Attention, gelu
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model

class Prediction:
    def __init__(self, path_model='modelos\26_12_2024'):

        K.clear_session()
        
        custom_objects = {
            "Attention": Attention,          # Classe de atenção
            "GeluActivation": GeluActivation # Função de ativação personalizada
        }

        # Carregar o modelo
        self.modelo_carregado = load_model(path_model + '/modelo_atencao.h5', custom_objects=custom_objects)

        # Exibir o resumo do modelo carregado
        self.modelo_carregado.summary()

        # Carregar o tokenizer
        with open(path_model + "/tokenizer.pickle", "rb") as handle_file:
            self.tokenizer = pickle.load(handle_file)

        # Carregar as categorias
        with open(path_model + "/categorias.pickle", "rb") as handle_file:
            self.categorias = pickle.load(handle_file)

        # Define maxlen (deve ser igual ao usado no treinamento)
        self.maxlen = self.modelo_carregado.input_shape[1]
    
    def run_predict(self, sentence):
        """
        Prediz a classe para uma sentença de entrada.
        """
        text_to_predict = [sentence]  # Formata a entrada como lista
        # Tokeniza e ajusta o comprimento
        text_to_predict = self.tokenizer.texts_to_sequences(text_to_predict)
        text_to_predict = pad_sequences(text_to_predict, maxlen=self.maxlen, dtype="int32", value=0)
        
        # Realiza a predição
        prediction = self.modelo_carregado.predict(text_to_predict, batch_size=1)[0]
        
        # Retorna a predição
        return prediction
    
    def run_predict_all(self, sentences, batch_size=32):
        """
        Prediz as classes para múltiplas sentenças em batch.
        """
        # Tokeniza e ajusta o comprimento
        text_to_predict = self.tokenizer.texts_to_sequences(sentences)
        text_to_predict = pad_sequences(text_to_predict, maxlen=self.maxlen, dtype="int32", value=0)
        
        # Realiza a predição em lote
        predictions = self.modelo_carregado.predict(text_to_predict, batch_size=batch_size)
        
        # Retorna as predições
        return predictions
    
    def getCategorias(self):
        """
        Retorna as categorias associadas ao modelo.
        """
        return self.categorias

        