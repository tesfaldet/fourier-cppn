import tensorflow as tf
import numpy as np


def IDFTLayer(name, input_coord, fourier_coord, coefficients):
    with tf.name_scope(name):
        input_shape = tf.shape(input_coord)
        batch_size = input_shape[0]
        input_width = input_shape[2]
        input_height = input_shape[1]

        fourier_shape = tf.shape(fourier_coord)
        fourier_width = fourier_shape[2]
        fourier_height = fourier_shape[1]

        # Reshape input coord to B x (H x W) x 2
        xy_coord = tf.reshape(input_coord,
                              [batch_size, input_width * input_height, 2])

        # Reshape fourier coord to B x (H_f x W_f) x 2
        f_coord = \
            tf.reshape(fourier_coord,
                       [batch_size, fourier_width * fourier_height, 2])

        # Transpose fourier coord to B x 2 x (H_f x W_f)
        f_coord_t = tf.transpose(f_coord, [0, 2, 1])

        # Normalize with fourier coord width and height
        norm = tf.to_float([[fourier_width], [fourier_height]])  # 2 x 1
        norm = tf.expand_dims(norm, axis=0)  # 1 x 2 x 1
        f_coord_t = f_coord_t / norm

        # Matrix multiply input coord with fourier coord
        # B x (H x W) x (H_f x W_f)
        xyf_coord = tf.matmul(xy_coord, f_coord_t)

        # Reshape to B x H x W x (H_f x W_f)
        xyf_coord = tf.reshape(xyf_coord,
                               [batch_size, input_height, input_width, -1])

        # Fourier sinusoidal basis B x H x W x (H_f x W_f)
        f_basis = tf.dtypes.complex(0., 2 * np.pi * xyf_coord)
        f_basis = tf.exp(f_basis)

        # Combine basis with fourier coefficients B x H x W x 1
        output = \
            tf.reduce_sum(coefficients * f_basis, axis=-1, keep_dims=True)

        # Normalize and take real part
        output = \
            tf.real(output) / \
            tf.sqrt(tf.cast(fourier_width * fourier_height, tf.float32))

        return output
