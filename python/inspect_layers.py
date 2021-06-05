import os
from lime import lime_image
import skimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, AveragePooling2D, \
                                    GlobalMaxPooling2D, Activation, Lambda, Flatten, \
                                    Dropout, BatchNormalization, SeparableConv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
#from tensorflow_addons.losses import TripletHardLoss, TripletSemiHardLoss
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from t_config import *
from model import get_model


############################################################################################################

model = get_model()
model.load_weights(TF_H5_FILEPATH)

CHANNELS = 3
############################################################################################################
DIRECTORY = "../../../Datasets/mask_dataset"
i0 = os.path.join(DIRECTORY, FILTER_FOLDER[0],f'{FILTER_FOLDER[0]}_016.jpg')
i1 = os.path.join(DIRECTORY, FILTER_FOLDER[1],f'{FILTER_FOLDER[1]}_023.jpg')

def to_int8(input_img):
    input_arr = input_img - 127
    input_arr = input_arr.astype(DTYPE)
    return input_arr

def get_image(ifolder):
    image = tf.keras.preprocessing.image.load_img(
        ifolder, color_mode='rgb' if CHANNELS == 3 else 'grayscale', target_size=TARGET_SIZE,
        interpolation='nearest'
    )
    #display(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    return np.array([input_arr], dtype=np.int16)




############################################################################################################

#for k,l in enumerate([layer for layer in model.layers if 'conv' in layer.name.lower()]):
#    inspect_conv = Model(model.inputs, l.output)
#    for conv_img in [i1, i0]:
#        convolutions = inspect_conv(get_image(conv_img))
#        plt.figure(figsize=(14, 12))
#        imgs_per_row = 4
#        for ii in range(convolutions.shape[3]):
#            plt.subplot((convolutions.shape[3] + 1) // imgs_per_row , imgs_per_row, ii+1).imshow(convolutions[0,...,ii], cmap='Greys_r')
#            plt.axis('off')
#
#        plt.tight_layout()
#        plt.title(f"Result After Convolution {k}")
#        plt.show()

############################################################################################################
img_0 = get_image(i0)[0]
img_1 = get_image(i1)[0]

def predict_fn(in_r):
    in_r = in_r * np.array([[0.2125,0.7154, 0.0721]])
    r = np.array(in_r.sum(axis=-1)[..., np.newaxis] - 127.).astype(np.int8)
    return model.predict(r)

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img_0, predict_fn, top_labels=1, num_samples=1000)
_, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
plt.imshow(img_0 * np.expand_dims(mask, axis=-1))
plt.axis('off')
plt.show()

explanation = explainer.explain_instance(img_1, predict_fn, top_labels=1, num_samples=1000)
_, mask = explanation.get_image_and_mask(explanation.top_labels[0],positive_only=True,num_features=2,hide_rest=False)
plt.imshow(img_1 * np.expand_dims(mask, axis=-1))
plt.axis('off')
plt.show()

############################################################################################################
