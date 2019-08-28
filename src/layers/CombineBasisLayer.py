import tensorflow as tf


def CombineBasisLayer(name, features_r, features_g, features_b,
                      shifted_image_basis):
    with tf.name_scope(name):
        cppn_num_neurons = tf.shape(shifted_image_basis)[1]
        fourier_basis_size = \
            tf.cast(tf.shape(shifted_image_basis)[2] / 2, tf.int32)
        batch_size = tf.shape(features_r)[0]
        input_height = tf.shape(features_r)[1]
        input_width = tf.shape(features_r)[2]

        # (B x H x W) x cppn_num_neurons
        features_r = tf.reshape(features_r, [-1, cppn_num_neurons])
        features_g = tf.reshape(features_g, [-1, cppn_num_neurons])
        features_b = tf.reshape(features_b, [-1, cppn_num_neurons])

        # (B x H x W) x cppn_num_neurons x 1
        features_r = tf.expand_dims(features_r, -1)
        features_g = tf.expand_dims(features_g, -1)
        features_b = tf.expand_dims(features_b, -1)

        # (B x H x W) x (fourier_basis_size x 2)
        coeffs_r = tf.reduce_sum(features_r * shifted_image_basis, axis=1)
        coeffs_g = tf.reduce_sum(features_g * shifted_image_basis, axis=1)
        coeffs_b = tf.reduce_sum(features_b * shifted_image_basis, axis=1)

        # B x H x W x (fourier_basis_size x 2)
        coeffs_r = tf.reshape(coeffs_r, [batch_size, input_height, input_width,
                                         fourier_basis_size * 2])
        coeffs_g = tf.reshape(coeffs_g, [batch_size, input_height, input_width,
                                         fourier_basis_size * 2])
        coeffs_b = tf.reshape(coeffs_b, [batch_size, input_height, input_width,
                                         fourier_basis_size * 2])

        # Make complex
        # B x H x W x fourier_basis_size
        coeffs_r = \
            tf.dtypes.complex(
                coeffs_r[..., :fourier_basis_size],
                coeffs_r[..., fourier_basis_size:])
        coeffs_g = \
            tf.dtypes.complex(
                coeffs_g[..., :fourier_basis_size],
                coeffs_g[..., fourier_basis_size:])
        coeffs_b = \
            tf.dtypes.complex(
                coeffs_b[..., :fourier_basis_size],
                coeffs_b[..., fourier_basis_size:])

        return coeffs_r, coeffs_g, coeffs_b
