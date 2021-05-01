from BEGANv1 import BEGAN
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

gan = BEGAN(is_test=True)

weights_path = './RESULTS/gen_epoch_57600.h5'
generator = gan.generator
generator.load_weights(weights_path)

n_rows = 10
n_cols = 10

noise_sample = np.random.normal(0, 1, (n_rows * n_cols, gan.noise_input))
generated = generator.predict(noise_sample)

fig, axs = plt.subplots(n_rows, n_cols)
for x in range(n_cols):
    for y in range(n_rows):
        img = (generated[y+x*n_cols, :, :] + 0.5) * 255.
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img[:, :, ::-1])
        axs[y, x].imshow(img)
        axs[y, x].axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
