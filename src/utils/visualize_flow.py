import tensorflow as tf
import numpy as np
from src.utils.cart_to_polar import cart_to_polar
from src.utils.rescale_values import rescale_values


def visualize_flow(flow, norm=False):
    fx, fy = flow[..., 0], flow[..., 1]
    v, ang = cart_to_polar(fx, fy)  # returns angle in rads

    # hsv_to_rgb expects everything to be in range [0, 1]
    h = ang / (2 * np.pi)
    s = tf.ones_like(h)

    if norm:
        v = rescale_values(v)
    else:
        v = tf.clip_by_value(v / 10.0, 0.0, 1.0)

    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255

    return tf.cast(rgb, tf.uint8)
