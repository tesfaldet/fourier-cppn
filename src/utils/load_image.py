import numpy as np
from scipy.misc import imread


def load_image(path):
    im = imread(path).astype('float32')  # [0, 255]
    return np.expand_dims(im, 3) / 255.0  # [0, 1]
