from cv2 import imread
import numpy as np


def load_image(path):
    im = imread(path).astype('float32')  # [0, 255]
    im = im[..., ::-1]  # BGR -> RGB
    im = np.expand_dims(im, 0)  # add batch dimension
    return im / 255.0  # [0, 1]
