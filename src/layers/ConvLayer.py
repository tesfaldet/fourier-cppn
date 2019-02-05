import tensorflow as tf


def ConvLayer(name, input, out_channels, ksize=1, stride=1,
              activation='relu', trainable=True):
    in_channels = input.get_shape().as_list()[3]
    shape_in = [ksize, ksize, in_channels, out_channels]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weight',
                            initializer=tf.initializers.random_uniform(-0.5, 0.5),
                            shape=shape_in, trainable=trainable)
        b = tf.get_variable('bias', initializer=tf.constant(0.0,
                            shape=[out_channels], dtype=tf.float32),
                            trainable=trainable)
        y = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1],
                         padding='SAME') + b

        if activation is not None:
            y = tf.nn.relu(y)

        return y