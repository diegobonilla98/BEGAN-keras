from BEGANv1 import BEGAN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K


def on_trackbar(val):
    pass


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

gan = BEGAN(is_test=True)

weights_path = './RESULTS/gen_epoch_57600.h5'
generator = gan.generator
generator.load_weights(weights_path)

noise_sample = np.random.normal(0, 1, (1, gan.noise_input))

cv2.namedWindow('Result', cv2.WINDOW_KEEPRATIO)
for i in range(16):
    cv2.createTrackbar(f'Val{i}', 'Result', 0, 100, on_trackbar)

while True:
    noise = (np.array([cv2.getTrackbarPos(f'Val{i}', 'Result') for i in range(16)]) / 50) - 1
    noise_sample[:, :] = np.tile(noise, 4)
    generated = generator.predict(noise_sample)[0]
    generated = np.uint8((generated + 0.5) * 255.)
    cv2.imshow('Result', generated)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
