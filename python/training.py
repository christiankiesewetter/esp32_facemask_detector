import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalCrossentropy
#from tensorflow_addons.losses import TripletHardLoss, TripletSemiHardLoss
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from t_config import *
from model import get_model

VISUALIZE = True
DIRECTORY = "../../../Datasets/mask_data_III"
FILTER_FOLDER = ['without_mask','with_mask']
# EPOCHS = 1
# steps_per_epoch = 6
# validation_steps = 2

image_gen_args = dict(
    validation_split=VAL_SPLIT,
    rescale=1/255.,
)

flow_args = dict(
    directory=DIRECTORY,
    target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
    classes=FILTER_FOLDER,
    class_mode='categorical',
    color_mode= 'rgb' if CHANNELS == 3 else 'grayscale',
    batch_size=1,
    shuffle=True,
    seed=42,
    save_format='jpeg',
)

augment_args = dict(
    rotation_range=25,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=(0.3, 2),
    shear_range=0.6,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
)

FINAL_SHAPE = tf.TensorShape([*TARGET_SIZE, CHANNELS])
output_types = ((tf.float32), (tf.float32))
output_shapes = ((TARGET_SIZE[0], TARGET_SIZE[1], CHANNELS),(1,))


filecount = 0
for folder in FILTER_FOLDER:
    filecount += [len(f) for a,d,f in os.walk(os.path.join(DIRECTORY, folder))][0]
steps_per_epoch = int(filecount * (1-VAL_SPLIT)) // BATCH_SIZE
validation_steps = int(filecount * VAL_SPLIT) // BATCH_SIZE


def get_dss():

    @tf.function
    def squeeze(features, label):
        return tf.reshape(tf.squeeze(features), FINAL_SHAPE), [tf.argmax(tf.squeeze(label))]

    img_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        **image_gen_args,
        **augment_args
    )
    data_gen_train = (squeeze(feature, label) for feature, label in img_gen_train.flow_from_directory(
        subset='training',
        **flow_args
    ))


    ds_train = tf.data.Dataset.from_generator(
        lambda: data_gen_train,
        output_types = output_types,
        output_shapes = output_shapes,
    )

    img_gen_valid = tf.keras.preprocessing.image.ImageDataGenerator(
        **image_gen_args
    )
    data_gen_valid = (squeeze(feature, label) for feature, label in img_gen_valid.flow_from_directory(
        subset='validation',
        **flow_args
    ))

    ds_valid = tf.data.Dataset.from_generator(
        lambda: data_gen_valid,
        output_types = output_types,
        output_shapes = output_shapes
    )

    if VISUALIZE:
        for d in ds_train.batch(BATCH_SIZE).take(1):
            images, labels = d[0].numpy(), d[1].numpy()
            plt.figure(figsize=(16, images.shape[0]))
            for ii in range(images.shape[0]):
                ax = plt.subplot(int(images.shape[0]/4), 4, ii+1)
                ax.imshow(images[ii] , cmap='Greys_r')
                plt.axis('off')
                ax.set_title(labels[ii])

            plt.tight_layout()
            plt.show()
            print(images.shape)

    return ds_train, ds_valid


def build_model():
    model = get_model()
    estop = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        min_delta=1e-5,
        patience=21,
        verbose=0,
        restore_best_weights=True)

    model.compile(
        optimizer=Adam(1e-04),
        loss = 'mse',
        metrics = 'binary_accuracy')

    callbacks = [estop]
    return model, callbacks


if __name__ == '__main__':
    model, callbacks = build_model()

    ds_train, ds_valid = get_dss()

    model.fit(
        x=ds_train.batch(BATCH_SIZE, drop_remainder=True),
        validation_data=ds_valid.batch(BATCH_SIZE, drop_remainder=True),
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks)

    model.save_weights(TF_H5_FILEPATH)

    ############## After Training Quantization ################
    ####### Therefore we need a representative dataset ########
    ###########################################################
    def representative_dataset():
        for data, labels in ds_valid.batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quantized_model = converter.convert()
    open(TFLITE_FILE_PATH, "wb").write(tflite_quantized_model)
