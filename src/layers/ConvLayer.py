import tensorflow as tf


def ConvLayer(name, input, out_channels, in_channels=None,
              ksize=1, stride=1,
              activation='relu',
              weight_init=tf.initializers.zeros(),
              bias_init=tf.initializers.zeros(),
              trainable=True):
    if in_channels is None:
        in_channels = input.shape[-1]
    shape = [ksize, ksize, in_channels, out_channels]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weight',
                            initializer=weight_init,
                            shape=shape,
                            trainable=trainable)
        if bias_init is None:
            y = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1],
                             padding='SAME')
        else:
            b = tf.get_variable('bias',
                                initializer=bias_init,
                                shape=[out_channels], dtype=tf.float32,
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
        elif activation == 'tanh2':
            y = tf.nn.tanh(y) ** 2

        return y
