import tensorflow as tf


def create_meshgrid(width, height, minval_x=-1.,
                    maxval_x=1., minval_y=-1., maxval_y=1.):
    minval_x = tf.cast(minval_x, tf.float32)
    maxval_x = tf.cast(maxval_x, tf.float32)
    minval_y = tf.cast(minval_y, tf.float32)
    maxval_y = tf.cast(maxval_y, tf.float32)
    x_coords, y_coords = tf.meshgrid(tf.linspace(minval_x, maxval_x, width),
                                     tf.linspace(minval_y, maxval_y, height))
    xy_coords = tf.cast(tf.stack([x_coords, y_coords], 2),
                        tf.float32)  # [height, width, 2]
    xy_coords = tf.expand_dims(xy_coords, axis=0)  # [1, height, width, 2]
    return xy_coords  # +x right, +y down