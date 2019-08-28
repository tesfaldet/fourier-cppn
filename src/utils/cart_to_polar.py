import tensorflow as tf
import numpy as np
from src.utils.atan2 import atan2


def cart_to_polar(x, y, degrees=False):
    v = tf.sqrt(x ** 2 + y ** 2)
    ang = atan2(y, x)
    scale = 1.0 if degrees else np.pi / 180.0
    return v, ang * scale
