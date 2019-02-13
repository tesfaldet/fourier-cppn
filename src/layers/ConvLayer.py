import tensorflow as tf


def ConvLayer(name, input, out_channels, ksize=1, stride=1,
              activation='relu', zeros=False, trainable=True):
    in_channels = input.get_shape().as_list()[3]
    shape_in = [ksize, ksize, in_channels, out_channels]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if zeros is True:
            init = tf.initializers.zeros()
        else:
            init = tf.initializers.random_normal(0, tf.sqrt(1.0/out_channels))
        w = tf.get_variable('weight',
                            initializer=init,
                            shape=shape_in, trainable=trainable)
        b = tf.get_variable('bias', initializer=tf.constant(0.0,
                            shape=[out_channels], dtype=tf.float32),
                            trainable=trainable)
        y = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1],
                         padding='SAME') + b

        if activation == 'relu':
            y = tf.nn.relu(y)
        elif activation == 'atan':
            y = tf.atan(y)
            y = tf.concat([y/0.67, (y * y)/0.6], -1)
        elif activation == 'sigmoid':
            y = tf.nn.sigmoid(y)

        return y