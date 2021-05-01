from BEGANv1 import BEGAN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)


def getEquidistantPoints(p1, p2, n):
    return [[lerp(p1[j], p2[j], 1. / n * i) for j in range(gan.noise_input)] for i in range(n + 1)]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

gan = BEGAN(is_test=True)

weights_path = './RESULTS/gen_epoch_57600.h5'
generator = gan.generator
generator.load_weights(weights_path)

n_points = 5
noise_upper = np.random.normal(0, 1, (gan.noise_input,))
noise_lower = np.random.normal(0, 1, (gan.noise_input,))

noise_sample = np.array(getEquidistantPoints(noise_lower, noise_upper, n_points))
generated = generator.predict(noise_sample)

fig, axs = plt.subplots(1, n_points + 1)
for n in range(n_points + 1):
    img = (generated[n] + 0.5) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    img = np.uint8(img[:, :, ::-1])
    axs[n].imshow(img)
    axs[n].axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
