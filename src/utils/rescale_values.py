import tensorflow as tf


def rescale_values(input, new_min=0., new_max=1., axis=None):
    return \
        ((input - tf.reduce_min(input, axis=axis)) * (new_max - new_min)) /\
        (tf.reduce_max(input, axis=axis) - tf.reduce_min(input, axis=axis))
