import tensorflow as tf
import numpy as np


def ApplyShiftsLayer(name, shifts, complex_image_basis, fourier_coord):
    with tf.name_scope(name):
        cppn_num_neurons = tf.shape(complex_image_basis)[0]
        fourier_height = tf.shape(fourier_coord)[1]
        fourier_width = tf.shape(fourier_coord)[2]
        fourier_basis_size = fourier_height * fourier_width
        batch_size = tf.shape(shifts)[0]
        input_height = tf.shape(shifts)[1]
        input_width = tf.shape(shifts)[2]

        # B x (H x W) x cppn_num_neurons x 2
        shifts = tf.reshape(shifts, [batch_size, input_height * input_width,
                                     cppn_num_neurons, 2])

        # B x cppn_num_neurons x (H x W) x 2
        shifts = tf.transpose(shifts, [0, 2, 1, 3])

        # FOURIER COORD
        # B x H_f x W_f x 2 -> B x fourier_basis_size x 2
        f_coord = tf.reshape(fourier_coord,
                             [batch_size, fourier_basis_size, 2])

        # B x 2 x fourier_basis_size
        f_coord = tf.transpose(f_coord, [0, 2, 1])

        # Normalize with fourier coord width and height
        norm = tf.to_float([[fourier_width], [fourier_height]])  # 2 x 1
        norm = tf.expand_dims(norm, axis=0)  # 1 x 2 x 1
        f_coord = f_coord / norm

        # B x 1 x 2 x fourier_basis_size
        f_coord = tf.expand_dims(f_coord, 1)

        # B x cppn_num_neurons x 2 x fourier_basis_size
        f_coord = tf.tile(f_coord, [1, cppn_num_neurons, 1, 1])

        # Matrix multiply shifts with fourier coord
        # B x cppn_num_neurons x (H x W) x fourier_basis_size
        shifted_f_coord = tf.matmul(shifts, f_coord)

        # Fourier shift
        # (https://en.wikipedia.org/wiki/Multidimensional_transform#Shift)
        # B x cppn_num_neurons x (H x W) x fourier_basis_size
        f_shift = tf.dtypes.complex(0., -2 * np.pi * shifted_f_coord)
        f_shift = tf.exp(f_shift)

        # Reshape to B x cppn_num_neurons x H x W x fourier_basis_size
        f_shift = tf.reshape(f_shift, [batch_size, cppn_num_neurons,
                                       input_height, input_width,
                                       fourier_basis_size])

        # B x H x W x cppn_num_neurons x fourier_basis_size
        f_shift = tf.transpose(f_shift, [0, 2, 3, 1, 4])

        # (B x H x W) x cppn_num_neurons x fourier_basis_size
        f_shift = tf.reshape(f_shift, [-1, cppn_num_neurons,
                                       fourier_basis_size])

        # SHIFT IMAGE BASIS
        # 1 x cppn_num_neurons x fourier_basis_size
        complex_image_basis = tf.expand_dims(complex_image_basis, 0)

        # (B x H x W) x cppn_num_neurons x fourier_basis_size
        shifted_complex_image_basis = f_shift * complex_image_basis

        # Make real
        # (B x H x W) x cppn_num_neurons x (fourier_basis_size x 2)
        shifted_image_basis = \
            tf.concat([tf.real(shifted_complex_image_basis),
                       tf.imag(shifted_complex_image_basis)], -1)

        return shifted_image_basis
