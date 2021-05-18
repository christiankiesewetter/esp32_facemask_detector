import os
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

############################################################################################################
DIRECTORY = "../../../Datasets/mask_dataset"
i0 = os.path.join(DIRECTORY, FILTER_FOLDER[0],f'{FILTER_FOLDER[0]}_010.jpg')
i1 = os.path.join(DIRECTORY, FILTER_FOLDER[1],f'{FILTER_FOLDER[1]}_011.jpg')

def get_image(ifolder):
    image = tf.keras.preprocessing.image.load_img(
        ifolder, color_mode='rgb' if CHANNELS == 3 else 'grayscale', target_size=TARGET_SIZE,
        interpolation='nearest'
    )
    #display(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    return input_arr.astype(DTYPE)

############################################################################################################


for k,l in enumerate([layer for layer in model.layers if 'conv' in layer.name.lower()]):
    inspect_conv = Model(model.inputs, l.output)
    for conv_img in [i1, i0]:
        convolutions = inspect_conv(get_image(conv_img))
        plt.figure(figsize=(14, 12))
        imgs_per_row = 4
        for ii in range(convolutions.shape[3]):
            plt.subplot((convolutions.shape[3] + 1) // imgs_per_row , imgs_per_row, ii+1).imshow(convolutions[0,...,ii], cmap='Greys_r')
            plt.axis('off')

        plt.tight_layout()
        plt.title(f"Result After Convolution {k}")
        plt.show()
