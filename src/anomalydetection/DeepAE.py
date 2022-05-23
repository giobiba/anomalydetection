import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
import pandas as pd
import numpy as np
import os
import json

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01


class DAE(tf.keras.Model):

    params_file = 'params.json'
    weights_path = 'weights'

    def __init__(self,  encoder_layer_size=None, decoder_layer_size=None,
                 activation_function='tanh',
                 regularizer='l2', regularizer_penalty=0.1,
                 dropout=0.5, threshold=None, contamination=0.1):
        super().__init__()

        self.labels_ = None
        self.decision_scores_ = None
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.activation_function = activation_function
        self.regularizer = regularizer
        self.regularizer_penalty = regularizer_penalty
        self.dropout = dropout

        self.threshold = threshold
        self.contamination = contamination

        self._create_model()

    def _create_model(self):
        # encoder layer
        if self.regularizer == "l1":
            from tensorflow.keras.regularizers import L1
            self.activity_regularizer = L1(self.regularizer_penalty)
        elif self.regularizer == "l2":
            from tensorflow.keras.regularizers import L2
            self.activity_regularizer = L2(self.regularizer_penalty)

        self.encoder = tf.keras.Sequential([
            Input(shape=self.encoder_layer_size[0])
        ])
        for shape in self.encoder_layer_size:
            self.encoder.add(Dense(units=shape,
                                   activation=self.activation_function,
                                   activity_regularizer=self.activity_regularizer))
            self.encoder.add(Dropout(self.dropout))

        # decoder layer
        self.decoder = tf.keras.Sequential()
        for shape in self.decoder_layer_size:
            self.decoder.add(Dense(units=shape,
                                   activation=self.activation_function,
                                   activity_regularizer=self.activity_regularizer))
            self.decoder.add(Dropout(self.dropout))

    def call(self, inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)

    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    def predict_score(self, x):
        x_prime = super().predict(x)
        self.decision_scores_ = np.log(np.linalg.norm(x - x_prime, ord=2, axis=1))

        if self.threshold is None:
            self.threshold = pd.Series(self.decision_scores_).quantile(1 - self.contamination)

        self.labels_ = (self.decision_scores_ > self.threshold).astype(int)

        return self.decision_scores_, self.labels_

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.save_weights(os.path.join(path, DAE.weights_path), save_format='tf')

        params = {
            'history_': self.history_,
            'decision_scores_': self.decision_scores_,
            'labels_': self.labels_,
            'threshhold_': self.threshold_,
            'contamination': self.contamination
        }
        open(os.path.join(path, (DAE.params_file), "w")).write(json.dumps(params))


    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist")
