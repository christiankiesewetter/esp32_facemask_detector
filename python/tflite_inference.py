import tensorflow as tf
import sys
import os
import numpy as np
from t_config import *


interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
    return np.squeeze(output_data)


if __name__ == "__main__":
    import numpy as np
    import cv2

    np.set_printoptions(threshold=sys.maxsize)#print(res.min(), res.max(), res.shape)
    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96,96))
        gray = np.array(gray).astype(np.int8)[np.newaxis,...,np.newaxis]

        res = predict(gray)
        print(f'Face Uncovered: {res[1]}, Face Masked: {res[0]}')

        cv2.imshow('frame', gray.squeeze().astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
