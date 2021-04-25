import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, AveragePooling2D, \
                                    GlobalMaxPooling2D, Activation, Lambda, Flatten, \
                                    Dropout, BatchNormalization, SeparableConv2D, MaxPooling2D
from t_config import *

def get_model():
    ### Model Creation ###
    model = Sequential([
        Conv2D(16, 3, activation='relu', padding='valid', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], CHANNELS)),
        MaxPooling2D(2,2),
        Conv2D(16, 3, activation='relu',padding='valid'),
        MaxPooling2D(2,2),
        Conv2D(16, 3, activation='relu',padding='valid'),
        MaxPooling2D(2,2),
        Conv2D(16, 3, activation='relu',padding='valid'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    print(model.summary())

    return model


if __name__ == '__main__':
    get_model()
