import numpy as np


VGG_MEAN = np.array([123.68, 116.779, 103.939],
                    dtype='float32').reshape((1, 1, 3))  # RGB


def vgg_process(image):
    image = image * 255.0  # [0,1] -> [0,255]
    image = image - VGG_MEAN  # mean subtract
    image = image[..., ::-1]  # RGB -> BGR
    return image


def vgg_deprocess(image, no_clip=False, unit_scale=False):
    image = image[..., ::-1]  # BGR -> RGB
    image = image + VGG_MEAN
    if not no_clip:
        image = np.clip(image, 0, 255).astype('uint8')
    if unit_scale:
        image = image / 255.0
    return image
