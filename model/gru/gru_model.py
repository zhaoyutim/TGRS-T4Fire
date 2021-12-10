import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K
class GRUModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        # self.gru_layers=gru_layers
        self.num_classes = num_classes
        # self.projection_dims=projection_dims
        # self.model = self.get_model(activation)
        self.model_10_layers = self.get_model_10_layers(input_shape, num_classes)

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    def r2_keras(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def get_model(self, input_shape, num_class):
        gru_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(256, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.Dense(num_class, activation='sigmoid')
        ])
        return gru_model

    def get_model_10_layers(self, input_shape, num_class):
        gru_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(256, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.GRU(256, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(128, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(128, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(64, dropout=0.2, return_sequences=True),
            tf.keras.layers.GRU(32, dropout=0.2,return_sequences=True),
            tf.keras.layers.GRU(32, dropout=0.2,return_sequences=True),
            tf.keras.layers.GRU(16, dropout=0.2,return_sequences=True),
            tf.keras.layers.GRU(16, dropout=0.2,return_sequences=True),
            tf.keras.layers.Dense(num_class, activation='sigmoid')
        ])
        return gru_model