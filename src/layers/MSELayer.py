import tensorflow as tf


def MSELayer(predicted, target):
    diffs = (predicted - target)**2
    avg = tf.reduce_mean(diffs)
    return avg