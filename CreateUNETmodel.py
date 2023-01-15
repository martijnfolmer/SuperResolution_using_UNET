from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Activation, Conv2DTranspose, \
    BatchNormalization


'''
    This class creates a machine learning model with the UNET architecture. This is a architecture that can
    be used for SuperResolution, denoising, semantic segmentation and other applications where the input and
    output can both be represented by an image.
    
    Note : the "small" unet is roughly 7M parameters in size, whilst the large one is 31M parameters in size. 
    Please take caution when training the large one, as you will run out of memory quickly with the 31M model.

    Author :        Martijn Folmer 
    Date created :  13-01-2023
'''


class CreateUNET:

    def __init__(self, input_shape, createBigUNET=False):
        self.createBigNet = createBigUNET       # if set to true, we create the bigger UNET
        self.input_shape = input_shape

    def convolution_block(self, input_layer, num_filters):
        """

        :param input_layer: The layer that enters this block
        :param num_filters: The number of filters of this block
        :return: The last layer of the block
        """
        x = Conv2D(num_filters, 3, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoder_block(self, input_layer, num_filters):
        """

        :param input_layer: The layer that enters this block
        :param num_filters: The number of filters of this block
        :return: the last layer of this block
        """
        x = self.convolution_block(input_layer, num_filters)
        pool_layer = MaxPool2D((2, 2))(x)
        return x, pool_layer

    def decoder_block(self, input_layer, skip_features, num_filters):
        """

        :param input_layer: The layer that enters this block
        :param skip_features: The layer we concatenate with
        :param num_filters: The number of filters of this block
        :return: The last layer of this block
        """
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_layer)
        x = Concatenate()([x, skip_features])
        x = self.convolution_block(x, num_filters)
        return x

    # Build Unet using the encoder and decoder blocks
    def get_model(self):
        inputs = Input(shape=self.input_shape)

        if self.createBigNet:
            e1, pool1 = self.encoder_block(inputs, 64)
            e2, pool2 = self.encoder_block(pool1, 128)
            e3, pool3 = self.encoder_block(pool2, 256)
            e4, pool4 = self.encoder_block(pool3, 512)

            bridge_layer = self.convolution_block(pool4, 1024)

            d1 = self.decoder_block(bridge_layer, e4, 512)
            d2 = self.decoder_block(d1, e3, 256)
            d3 = self.decoder_block(d2, e2, 128)
            d_final = self.decoder_block(d3, e1, 64)
        else:
            e1, pool1 = self.encoder_block(inputs, 64)
            e2, pool2 = self.encoder_block(pool1, 128)
            e3, pool3 = self.encoder_block(pool2, 256)

            bridge_layer = self.convolution_block(pool3, 512)

            d1 = self.decoder_block(bridge_layer, e3, 256)
            d2 = self.decoder_block(d1, e2, 128)
            d_final = self.decoder_block(d2, e1, 64)

        outputs = Conv2D(3, 1, padding="same", activation="relu")(d_final)  # Binary (can be multiclass)
        model = Model(inputs, outputs, name="UNET")

        model.summary()     # get the summary
        return model
