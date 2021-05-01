from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dense, Input, AveragePooling2D, Dropout, GaussianNoise, Flatten, Reshape, ELU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoader


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class BEGAN:
    def __init__(self):
        self.image_shape = (128, 128, 3)
        self.batch_size = 16
        self.noise_input = 64

        self.sample_X = np.random.normal(0, 1, (3, self.noise_input))

        self.dl = DataLoader(self.batch_size, self.image_shape)

        self.kt = np.float32(0.)
        self.lr = np.float32(1e-4)

        self.generator = self.create_generator()
        self.generator.summary()
        plot_model(self.generator, 'generator.png', show_shapes=True)

        self.encoder = self.create_encoder()
        self.encoder.summary()
        plot_model(self.encoder, 'encoder.png', show_shapes=True)

        self.decoder = self.create_decoder()
        self.decoder.summary()
        plot_model(self.decoder, 'decoder.png', show_shapes=True)

        input_tensor = Input(shape=self.image_shape)
        self.discriminator = Model(input_tensor, self.decoder(self.encoder(input_tensor)))
        self.discriminator.summary()
        plot_model(self.discriminator, 'discriminator.png', show_shapes=True)

        input_noise = Input(shape=(self.noise_input, ))
        input_image = Input(shape=self.image_shape)
        generated = self.generator(input_noise)
        d_real = self.discriminator(input_image)
        d_fake = self.discriminator(generated)

        self.kt_tensor = K.placeholder(shape=(1, ))
        self.d_real_loss = K.mean(K.abs(input_image - d_real))
        self.d_fake_loss = K.mean(K.abs(generated - d_fake))
        self.d_loss = self.d_real_loss - self.kt_tensor * self.d_fake_loss
        self.g_loss = self.d_fake_loss

        self.update_g = Adam(self.lr, 0.5).get_updates(self.g_loss, self.generator.trainable_weights)
        self.g_train = K.function([input_noise], [self.g_loss], self.update_g)

        self.update_dis = Adam(self.lr, 0.5).get_updates(self.d_loss, self.discriminator.trainable_weights)
        self.d_train = K.function([input_image, input_noise, self.kt_tensor],
                                  [self.d_real_loss, self.d_fake_loss, self.d_loss], self.update_dis)

    def create_generator(self):
        input_tensor = Input(shape=(self.noise_input, ))

        res = [8, 8, 64]
        x = Flatten()(input_tensor)
        x = Dense(np.prod(res), kernel_initializer=RandomNormal(0, 0.02))(x)
        x = Reshape(res)(x)

        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        output_tensor = Conv2D(3, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)

        return Model(input_tensor, output_tensor, name='generator')

    def create_encoder(self):
        input_tensor = Input(shape=self.image_shape)

        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(input_tensor)
        x = ELU()(x)

        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = Conv2D(128, 1, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = AveragePooling2D(strides=2, padding="same")(x)
        x = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(128, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = Conv2D(192, 1, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = AveragePooling2D(strides=2, padding="same")(x)
        x = Conv2D(192, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(192, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = Conv2D(256, 1, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = AveragePooling2D(strides=2, padding="same")(x)
        x = Conv2D(256, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(256, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = Conv2D(320, 1, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = AveragePooling2D(strides=2, padding="same")(x)
        x = Conv2D(320, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(320, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = Flatten()(x)
        output_tensor = Dense(64, kernel_initializer=RandomNormal(0, 0.02))(x)

        return Model(input_tensor, output_tensor, name='encoder')

    def create_decoder(self):
        input_tensor = Input(shape=(self.noise_input,))

        res = [8, 8, 64]
        x = Flatten()(input_tensor)
        x = Dense(np.prod(res), kernel_initializer=RandomNormal(0, 0.02))(x)
        x = Reshape(res, name='wtf')(x)

        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)
        x = Conv2D(64, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)
        x = ELU()(x)

        output_tensor = Conv2D(3, 3, strides=1, padding="same", kernel_initializer=RandomNormal(0, 0.02))(x)

        return Model(input_tensor, output_tensor, name='decoder')

    def plot_results(self, epoch):
        fake = self.generator.predict(self.sample_X)
        res = (np.hstack([fake[0, :, :, :], fake[1, :, :, :], fake[2, :, :, :]]) + 0.5) * 255.
        res[res > 255] = 255
        res[res < 0] = 0
        res = np.uint8(res[:, :, ::-1])
        plt.clf()
        # plt.figure(figsize=(10, 15))
        plt.imshow(res)
        plt.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def train(self, epochs):
        kt = np.float32(0.)
        gen_losses = []
        dis_losses = []
        for epoch in range(epochs):
            real = self.dl.load_batch()
            z = np.random.normal(0, 1, (self.batch_size, self.noise_input))
            fake = self.generator.predict(z)

            g_loss = self.g_train([z])
            d_real_loss, d_fake_loss, d_loss = self.d_train([real, z, kt])

            kt = np.maximum(np.minimum(1., kt + 0.001 * (0.5 * d_real_loss - d_fake_loss)), 0.)

            print(
                f"Epoch {epoch}/{epochs}:\t[Adv_loss: {g_loss}]\t[D_loss: {d_loss}], kt: {kt}")
            gen_losses.append(g_loss)
            dis_losses.append(d_loss)

            if epoch % 5 == 0:
                self.plot_results(epoch)
            if epoch % 100 == 0:
                self.generator.save_weights(f'./RESULTS/weights/gen_epoch_{epoch}.h5')
            if epoch % 2 == 0:
                plt.clf()
                plt.plot(gen_losses, label="Gen Loss", alpha=0.8)
                plt.plot(dis_losses, label="Dis Loss", alpha=0.2)
                plt.legend()
                plt.savefig(f'./RESULTS/metrics.png')
                plt.close()


gan = BEGAN()
gan.train(90_000)
