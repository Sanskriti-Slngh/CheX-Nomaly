from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, \
    Activation, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, Attention, add, Add, multiply, UpSampling2D, Lambda
from tensorflow.keras.models import Sequential
from model.base_model import BaseModel
from tensorflow.keras.models import Model as m


class Model(BaseModel):
    def __init__(self, history, data, disease, all, aug):
        self.name = 'unet'
        self.model = Sequential()
        self.batchnorm = False
        self.dropout = 0.0

        def conv_block(x, num_filters):
            x = Conv2D(num_filters, (3,3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, (3,3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        def attention_gate(g, s, filters):
            Wg = Conv2D(filters, (1,1), padding='same')(g)
            Wg = BatchNormalization()(Wg)

            Ws = Conv2D(filters, (1,1), padding="same")(s)
            Ws = BatchNormalization()(Ws)

            out = Activation("relu")(Wg + Ws)
            out = Conv2D(filters, (1,1), padding="same")(out)
            out = Activation("sigmoid")(out)

            return out * s

        def encoder_block(x, num_filters):
            x = conv_block(x, num_filters)
            p = MaxPooling2D((2, 2))(x)
            return x, p

        def decoder_block(x, s, num_filters):
            x = UpSampling2D(interpolation="bilinear")(x)
            s = attention_gate(x, s, num_filters)
            x = Concatenate()([x, s])
            x = conv_block(x, num_filters)
            return x

        inputs = Input(shape=(512, 512, 1))

        # Encoder path
        c1, p1 = encoder_block(inputs, 64)
        c2, p2 = encoder_block(p1, 128)
        c3, p3 = encoder_block(p2, 256)
        c4, p4 = encoder_block(p3, 512)

        c5 = conv_block(p4, 1024)

        d1 = decoder_block(c5, c4, 512)
        d2 = decoder_block(d1, c3, 256)
        d3 = decoder_block(d2, c2, 128)
        d4 = decoder_block(d3, c1, 64)

        # Final layer
        output = Conv2D(1, (1, 1), activation="sigmoid")(d4)

        # Model
        self.model = m(inputs=[inputs], outputs=output)

        print(self.model.summary())

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(data, disease, aug, all)