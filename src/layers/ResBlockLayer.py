import tensorflow as tf
from src.layers.ConvLayer import ConvLayer


def ResBlockLayer(name, input, out_channels,
                  activation='relu',
                  weight_init=tf.initializers.zeros(),
                  bias_init=tf.initializers.zeros(),
                  trainable=True):
    with tf.name_scope(name):
        layer_1 = ConvLayer(name + '/conv1', input,
                            out_channels=out_channels,
                            weight_init=weight_init,
                            activation='relu',
                            trainable=trainable)
        layer_2 = ConvLayer(name + '/conv2', layer_1,
                            out_channels=out_channels / 4,
                            weight_init=weight_init,
                            activation='relu',
                            trainable=trainable)
        layer_3 = ConvLayer(name + '/conv3', layer_2,
                            out_channels=out_channels / 4,
                            weight_init=weight_init,
                            activation='relu',
                            trainable=trainable)
        layer_4 = ConvLayer(name + '/conv4', layer_3,
                            out_channels=out_channels,
                            weight_init=weight_init,
                            activation=None,
                            trainable=trainable)
        y = layer_1 + layer_4

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
