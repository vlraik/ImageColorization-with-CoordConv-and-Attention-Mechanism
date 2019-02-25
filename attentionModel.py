import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, Activation, Dense, \
    Dropout, Flatten, Lambda, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers import Add, MaxPool2D, Concatenate, Embedding, RepeatVector
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
    v1 = Input(shape=(l, d))
    q1 = Input(shape=(l, d))
    k1 = Input(shape=(l, d))

    v2 = Dense(dv * nv, activation='relu')(v1)
    q2 = Dense(dv * nv, activation='relu')(q1)
    k2 = Dense(dv * nv, activation='relu')(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)

    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q, k])  # l, nv, nv
    att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)

    out = Add()([out, q1])

    out = Dense(dout, activation="relu")(out)

    return Model(inputs=[q1, k1, v1], outputs=out)




class _CoordinateChannel(Layer):
    """ Adds Coordinate Channels to the input tensor.
    # Arguments
        rank: An integer, the rank of the input data-uniform,
            e.g. "2" for 2D convolution.
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        ND tensor with shape:
        `(samples, channels, *)`
        if `data_format` is `"channels_first"`
        or ND tensor with shape:
        `(samples, *, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        ND tensor with shape:
        `(samples, channels + 2, *)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, *, channels + 2)`
        if `data_format` is `"channels_last"`.
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, rank,
                 use_radius=False,
                 data_format=None,
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)

        if data_format not in [None, 'channels_first', 'channels_last']:
            raise ValueError('`data_format` must be either "channels_last", "channels_first" '
                             'or None.')

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)

        if self.rank == 2:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

            if self.use_radius:
                rr = K.sqrt(K.square(xx_channels - 0.5) +
                            K.square(yy_channels - 0.5))
                outputs = K.concatenate([outputs, rr], axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CoordinateChannel2D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor.

    # Arguments
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, channels + 2/3, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels + 2/3)`
        if `data_format` is `"channels_last"`.

        If `use_radius` is set, then will have 3 additional filers,
        else only 2 additional filters will be added.

    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel2D, self).__init__(
            rank=2,
            use_radius=use_radius,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.pop('rank')
        return config

# Get the images
imageDataset = []
for filename in os.listdir('Opencountry/Train/'):
    imageDataset.append(image.img_to_array(image.load_img('Opencountry/Train/' + filename)))
imageDataset = np.array(imageDataset, dtype=float)

# Setting up the train and test split
split = int(1 * len(imageDataset))
Xtrain = imageDataset[:split]
Xtest = imageDataset[split:]
Xtrain = 1.0 / 255 * Xtrain

# Load the inception model
inception = InceptionResNetV2(weights='imagenet', include_top=True)
# inception.load_weights('/data/inception_weights.h5')
inception.graph = tf.get_default_graph()

# Implement the model
embed_input = Input(shape=(1000,))

print('Starting to load the main network after loading the inception pre-trained layer')

# Encoder
encoder_input = Input(shape=(256, 256, 1,))
#encoder_output = CoordinateChannel2D()(encoder_input)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
#encoder_output = BatchNormalization()(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

# Alternate attention before fusion layer
#Encoder output is 32 32 512

# attention_output = Reshape([32 * 32, 256 * 3])(encoder_output)
# att_vector = MultiHeadsAttModel(l=32 * 32, d=256 * 3, dv=16 * 3, dout=64, nv=16)
# attention_output = att_vector([attention_output, attention_output, attention_output])
# print(attention_output.shape)
# attention_output = Reshape([8 * 8, 1024])(attention_output)
# attention_output = BatchNormalization()(attention_output)

# Fusion
fusion_output = RepeatVector(32 * 32)(embed_input)
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = Concatenate(axis=3)([encoder_output, fusion_output])
encoder_output = BatchNormalization()(fusion_output)
fusion_output = Conv2D(256*3, (1, 1), activation='relu', padding='same')(fusion_output)

# Attention
# print(fusion_output.shape)
# attention_output = Reshape([32 * 32, 256 * 3])(fusion_output)
# att_vector = MultiHeadsAttModel(l=32 * 32, d=256 * 3, dv=16 * 3, dout=64, nv=16)
# attention_output = att_vector([attention_output, attention_output, attention_output])
# print(attention_output.shape)
# attention_output = Reshape([32,32, 64])(attention_output)
# attention_output = BatchNormalization()(attention_output)
# print(attention_output.shape)

# Decoder
decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal')(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='glorot_normal')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same',kernel_initializer='glorot_normal')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

print(model.summary())


# Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


# Image transformer
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True)

# Generate training data
batch_size = 15


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        X_batch = X_batch.reshape(X_batch.shape + (1,))
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

#print("Loading previous weights")
#model.load_weights("color_tensorflow_real_mode_og.h5")
# Train model
tensorboard = TensorBoard(log_dir="/output")
model.compile(optimizer='rmsprop', loss='mse')
checkpoint = ModelCheckpoint("weights.{epoch:02d}.hdf5", verbose=1,
                             save_best_only=True, mode=True)
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard, checkpoint], epochs=250, steps_per_epoch=30)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("color_tensorflow_real_mode_og.h5")

# Make predictions on validation images
# Change to '/data/images/Test/' to use all the 500 test images
color_me = []
for filename in os.listdir('Opencountry/Test/'):
    color_me.append(img_to_array(load_img('Opencountry/Test/' + filename)))
color_me = np.array(color_me, dtype=float)
color_me = 1.0 / 255 * color_me
color_me = gray2rgb(rgb2gray(color_me))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    imsave("resultimg_" + str(i) + ".png", lab2rgb(cur))
