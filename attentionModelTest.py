import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, Activation, Dense, \
    Dropout, Flatten, Lambda, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers import Add, MaxPool2D, Concatenate, Embedding, RepeatVector
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb, rgb2hsv,hsv2rgb
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
from keras.models import model_from_json


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



# Load the inception model
inception = InceptionResNetV2(weights='imagenet', include_top=True)
# inception.load_weights('/data/inception_weights.h5')
inception.graph = tf.get_default_graph()

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


# Make predictions on validation images
# Change to '/data/images/Test/' to use all the 500 test images
color_me = []
for filename in os.listdir('Opencountry/test/'):
    color_me.append(img_to_array(load_img('Opencountry/test/' + filename)))
color_me = np.array(color_me, dtype=float)
color_me = 1.0 / 255 * color_me
color_me = gray2rgb(rgb2gray(color_me))
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape + (1,))

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("color_tensorflow_real_mode_og.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='rmsprop', loss='mse')

# Test model
output = loaded_model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = color_me[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    print(output[i].shape)
    final = lab2rgb(cur)
    final = rgb2hsv(final)
    final[:,:,1] = final[:,:,1]*2
    np.putmask(final,final>255,255)
    final = hsv2rgb(final)
    imsave("resultimg_" + str(i) + ".png", final)
