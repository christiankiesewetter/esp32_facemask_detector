import tensorflow as tf

import tensorflow as tf
import sys
import os
import numpy as np
from t_config import *


interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def get_image(ifolder):
    image = tf.keras.preprocessing.image.load_img(
        ifolder, color_mode='rgb' if CHANNELS == 3 else 'grayscale', target_size=TARGET_SIZE,
        interpolation='nearest'
    )
    #display(image)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) - 128.
    return input_arr.astype(DTYPE)


def resize_image(inp, width, height, blocksize):
    ms = np.zeros((height//blocksize)*(width//blocksize), dtype=np.float32)
    ipos = 0
    for hstep in range(height//blocksize):
        for wstep in range(width//blocksize):
            val = 0.
            for jj in range(blocksize):
                for kk in range(blocksize):
                    regup = kk + (jj * width)
                    val += inp[(wstep * blocksize) + (hstep * blocksize * width) + regup]

            ms[ipos] = val / blocksize / blocksize
            ipos+=1

    ms = ms.reshape(height//blocksize, width//blocksize)
    return ms


def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return tf.squeeze(output_data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder_results = []
    for folder in FILTER_FOLDER:

        folder_results.append(
            [predict(get_image(os.path.join(DIRECTORY, folder, k))).numpy() for _,_,kl in os.walk(os.path.join(DIRECTORY, folder)) for k in kl]
        )


    print('Accuracy Facemask:\t{:0.2f}'.format(sum([1 for m in folder_results[0] if not m])/len(folder_results[0])))
    print('Accuracy No Mask:\t{:0.2f}'.format(sum([1 for m in folder_results[1] if m])/len(folder_results[1])))


    i0 = os.path.join(DIRECTORY, FILTER_FOLDER[0],f'{FILTER_FOLDER[0]}_022.jpg')
    i1 = os.path.join(DIRECTORY, FILTER_FOLDER[1],f'{FILTER_FOLDER[1]}_023.jpg')

    img = get_image(i0)
    plt.imshow(tf.squeeze(img), cmap="Greys_r");
    print(img.shape, img.min(), img.max())
    plt.axis('off')
    plt.title(f'Result:{predict(img)}')
    plt.show()
