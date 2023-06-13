from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential
from model.base_model import BaseModel
from tensorflow.keras.models import Model as m
import tensorflow as tf
from tensorflow.keras import layers
from transunet import TransUNet
from keras_unet_collection import models
from tensorflow import keras


class Model(BaseModel):
    def __init__(self, history, data, disease, all, aug):
        self.name = 'unet'
        self.model = Sequential()
        self.batchnorm = False
        self.dropout = 0.0

        # Define the input shape
        input_shape = (512, 512, 1)

        # Define the number of filters for each convolutional layer
        filters = [64, 128, 256, 512, 1024]

        # Define the number of attention heads for the transformer layers
        num_heads = [8, 8, 8, 8]

        # Define the number of transformer layers
        num_transformer_layers = 12

        # Define the activation function for the convolutional layers
        activation = 'relu'

        # Define the dropout rate for the convolutional layers
        dropout_rate = 0.5

        # Define if batch normalization
        batch_norm = False

        # Define the input layer
        inputs = keras.Input(shape=input_shape)

        # Downsample path
        x = inputs
        skip_connections = []
        for filter in filters[:-1]:
            # Convolutional block
            x = layers.Conv2D(filter, 3, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2D(filter, 3, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            skip_connections.append(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(dropout_rate)(x)

        # Bottom of the U-Net
        x = layers.Conv2D(filters[-1], 3, activation=activation, padding='same')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters[-1], 3, activation=activation, padding='same')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        # Upsample path
        for i, filter in enumerate(reversed(filters[:-1])):
            if batch_norm:
                x = layers.BatchNormalization()(x)

            # Convolutional block
            x = layers.Conv2D(filter, 3, activation=activation, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2D(filter, 3, activation=activation, padding='same')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.UpSampling2D(2)(x)

            # Attention block
            attention = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=filter // num_heads[i])
            query = layers.Conv2D(filter, 1)(x)
            value = layers.Conv2D(filter, 1)(skip_connections[-(i + 1)])
            key = attention(query, value)
            attention_output = layers.Conv2D(filter, 1, activation=activation)(key)

            x = layers.Concatenate()([attention_output, x])


        # Output layer
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

        # Define the model
        self.model = m(inputs=inputs, outputs=outputs)
        print(self.model.summary())


        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(data, disease, aug, all)