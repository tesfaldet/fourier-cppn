import tensorflow as tf


def stop_gradients(target, mask):
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target