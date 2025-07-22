import numpy as np
import tensorflow as tf
from keras import layers

class CNNTransformerNet :

    def positional_encoding(self, sequence_length, d_model):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc = np.zeros((sequence_length, d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return tf.cast(pos_enc, dtype=tf.float32)

    def transformer_encoder(self, inputs, num_heads=4, embed_dim=16, ff_dim=16, dropout_rate=0.1):
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
        attention = layers.Dropout(dropout_rate)(attention)
        attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        ff = layers.Dense(ff_dim, activation=tf.nn.swish)(attention)
        ff = layers.Dropout(dropout_rate)(ff)
        ff = layers.Dense(inputs.shape[-1])(ff)
        ff = layers.LayerNormalization(epsilon=1e-6)(attention + ff)
        return ff

    def nn_model(self, input_shape, parameter_type=None):
        input_layer = layers.Input(shape=input_shape)
        # Down sampling layers
        hidden_d = tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation=tf.nn.swish)(input_layer)
        hidden_d = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(hidden_d)
        hidden_d = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation=tf.nn.swish)(hidden_d)
        hidden_d = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(hidden_d)
        # Encoder
        pos_enc = CNNTransformerNet.positional_encoding(self,  hidden_d.shape[1], hidden_d.shape[2])
        hidden_d = hidden_d + pos_enc
        hidden_d = CNNTransformerNet.transformer_encoder(self, hidden_d)
        # Up sampling layers
        hidden_u = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation=tf.nn.swish)(hidden_d)
        hidden_u = tf.keras.layers.UpSampling1D(2)(hidden_u)
        hidden_u = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation=tf.nn.swish)(hidden_u)
        hidden_u = tf.keras.layers.UpSampling1D(2)(hidden_u)
        # Output BP waveform layer
        output_wave = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same', activation='linear')(hidden_u)
        model = tf.keras.Model(inputs=input_layer, outputs=output_wave)
        # Params
        if parameter_type == 'cp':
            params = tf.Variable(initial_value=np.array([0.0, 0.0, 0.0]), dtype=tf.float32, trainable=True)
        elif parameter_type == 'bbp':
            params = None
        elif parameter_type == 'pdp':
            params = tf.Variable(initial_value=np.array([0.0, 0.0, 0.0, 0.0]), dtype=tf.float32, trainable=True)
        else:
            params = None
        unc_param = tf.Variable(initial_value=np.array([2.0, 2.0]), trainable=True)
        return model, params, unc_param
