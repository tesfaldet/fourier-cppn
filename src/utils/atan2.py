import tensorflow as tf
import numpy as np


def atan2(y, x):
    # constants
    DBL_EPSILON = 2.2204460492503131e-16
    atan2_p1 = 0.9997878412794807 * (180 / np.pi)
    atan2_p3 = -0.3258083974640975 * (180 / np.pi)
    atan2_p5 = 0.1555786518463281 * (180 / np.pi)
    atan2_p7 = -0.04432655554792128 * (180 / np.pi)

    ax, ay = tf.abs(x), tf.abs(y)
    c = tf.where(tf.greater_equal(ax, ay), tf.div(ay, ax + DBL_EPSILON),
                 tf.div(ax, ay + DBL_EPSILON))
    c2 = tf.square(c)
    angle = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c
    angle = tf.where(tf.greater_equal(ax, ay), angle, 90.0 - angle)
    angle = tf.where(tf.less(x, 0.0), 180.0 - angle, angle)
    angle = tf.where(tf.less(y, 0.0), 360.0 - angle, angle)
    return angle
