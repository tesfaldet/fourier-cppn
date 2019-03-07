import tensorflow as tf


def ConvLayer(name, input, out_channels, ksize=1, stride=1,
              activation='relu',
              weight_init=tf.initializers.zeros(),
              trainable=True,
              no_bias=False,
              no_shape=False):
    in_channels = input.get_shape().as_list()[3]
    shape = [ksize, ksize, in_channels, out_channels]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if no_shape:
            w = tf.get_variable('weight',
                                initializer=weight_init,
                                trainable=trainable)
        else:
            w = tf.get_variable('weight',
                                initializer=weight_init,
                                shape=shape,
                                trainable=trainable)
        if no_bias:
            y = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1],
                             padding='SAME')
        else:
            b = tf.get_variable('bias', initializer=tf.constant(0.0,
                                shape=[out_channels], dtype=tf.float32),
                                trainable=trainable)
            y = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1],
                             padding='SAME') + b

        if activation == 'relu':
            y = tf.nn.relu(y)
        elif activation == 'atan_concat':
            y = tf.atan(y)
            y = tf.concat([y/0.67, (y * y)/0.6], -1)
        elif activation == 'atan':
            y = tf.atan(y)
        elif activation == 'sigmoid':
            y = tf.nn.sigmoid(y)
        elif activation == 'tanh':
            y = tf.nn.tanh(y)

        return y