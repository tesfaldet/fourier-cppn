from cv2 import imread, resize
import numpy as np


def load_image(path, size=None):
    im = imread(path).astype('float32')  # [0, 255]
    if size is not None:
        im = resize(im, dsize=size)
    im = im[..., ::-1]  # BGR -> RGB
    im = np.expand_dims(im, 0)  # add batch dimension
    return im / 255.0  # [0, 1]
