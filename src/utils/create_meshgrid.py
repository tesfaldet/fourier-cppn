import tensorflow as tf
import numpy as np


def create_meshgrid(width, height):
    x_coords, y_coords = tf.meshgrid(np.linspace(0, 1, width),
                                     np.linspace(0, 1, height))  # range 0-1
    xy_coords = tf.cast(tf.stack([x_coords, y_coords], 2),
                        tf.float32)  # [width, height, 2]
    xy_coords = tf.expand_dims(xy_coords, axis=0)  # [1, width, height, 2]
    return xy_coords  # +x right, +y down