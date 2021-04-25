import numpy as np

MODEL_NAME = "mask_model"

#TARGET_SIZE = (240, 320) #QVGA
TARGET_SIZE = (120, 160) #QQVGA
#TARGET_SIZE = (96, 96) #96x96

CHANNELS = 1
BATCH_SIZE = 16

DTYPE=np.int8

DIRECTORY = "../dataset"
FILTER_FOLDER = ['facemask','no_mask']

EPOCHS = 100
VAL_SPLIT = 0.2

TF_H5_FILEPATH = f"{MODEL_NAME}.h5"
TFLITE_FILE_PATH = f'{MODEL_NAME}.tflite'

VISUALIZE = False
