import cv2
import numpy as np
import glob
import os


class DataLoader:
    def __init__(self, batch_size, image_size):
        self.image_size = image_size[:2]
        self.batch_size = batch_size
        root_dir = '/media/bonilla/HDD_2TB_basura/databases/CelebA/archive/img_align_celeba/img_align_celeba'
        images_A = glob.glob(os.path.join(root_dir, '*.jpg'))
        root_dir = '/media/bonilla/HDD_2TB_basura/databases/UTKFace'
        images_B = glob.glob(os.path.join(root_dir, '*.jpg'))
        self.images = images_A + images_B

    def _load_image(self, path):
        image = cv2.imread(path)
        if path.split(os.sep)[-2] == 'img_align_celeba':
            image = image[35:180, 40:160, :]
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
        return image.astype(np.float32) / 255. - .5

    def load_batch(self):
        image_paths = np.random.choice(self.images, self.batch_size)
        X = [self._load_image(path) for path in image_paths]
        return np.array(X)


# dl = DataLoader(4, (256, 256, 3))
# dl.load_batch()