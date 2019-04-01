import tensorflow as tf
import numpy as np


def IDFTLayer(name, input_meshgrid, fourier_meshgrid, coefficients):
    with tf.name_scope(name):
        input_shape = input_meshgrid.get_shape().as_list()
        input_width = input_shape[2]
        input_height = input_shape[1]

        fourier_shape = fourier_meshgrid.get_shape().as_list()
        fourier_width = fourier_shape[2]
        fourier_height = fourier_shape[1]

        # Reshape input meshgrid to 1 x (H x W) x 2
        xy_meshgrid = tf.reshape(input_meshgrid,
                                 [1, input_width * input_height, 2])

        # Reshape fourier meshgrid to 1 x (H_f x W_f) x 2
        f_meshgrid = tf.reshape(fourier_meshgrid,
                                [1, fourier_width * fourier_height, 2])

        # Transpose fourier meshgrid to 1 x 2 x (H_f x W_f)
        f_meshgrid_t = tf.transpose(f_meshgrid, [0, 2, 1])

        # Normalize with fourier meshgrid width and height
        norm = np.array([[fourier_width], [fourier_height]],
                        dtype=np.float32)  # 2 x 1
        norm = tf.expand_dims(norm, axis=0)  # 1 x 2 x 1
        f_meshgrid_t = f_meshgrid_t / norm

        # Matrix multiply input meshgrid with fourier meshgrid
        # 1 x (H x W) x (H_f x W_f)
        xyf_meshgrid = tf.matmul(xy_meshgrid, f_meshgrid_t)

        # Reshape to 1 x H x W x (H_f x W_f)
        xyf_meshgrid = tf.reshape(xyf_meshgrid,
                                  [1, input_height, input_width, -1])

        # Fourier sinusoidal basis 1 x H x W x (H_f x W_f)
        f_basis = tf.dtypes.complex(0., 2 * np.pi * xyf_meshgrid)
        f_basis = tf.exp(f_basis)

        # Combine basis with fourier coefficients 1 x H x W x 1
        output = \
            tf.reduce_sum(coefficients * f_basis, axis=-1, keep_dims=True)

        # Normalize and take real part
        output = \
            tf.real(output) / tf.sqrt(tf.cast(fourier_width *
                                              fourier_height, tf.float32))

        return output