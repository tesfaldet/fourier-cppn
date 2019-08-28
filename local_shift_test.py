import tensorflow as tf
import cv2
import numpy as np
from src.layers.ApplyShiftsLayer import ApplyShiftsLayer
from src.layers.CombineBasisLayer import CombineBasisLayer
from src.layers.IDFTLayer import IDFTLayer
from src.utils.create_meshgrid import create_meshgrid


sess = tf.InteractiveSession()

# Input coord dimensions
input_height = 100
input_width = 100
batch_size = 1

# 1 x 100 x 100 x 2
input_coord = \
    create_meshgrid(input_width, input_height,
                    0, input_width - 1, 0, input_height - 1,
                    batch_size=tf.constant(batch_size))

# Fourier coord dimensions
f_height = 10
f_width = 10
fourier_basis_size = f_height * f_width
if f_height % 2 == 0:
    f_height_min = -(f_height / 2.0)
    f_height_max = (f_height / 2.0) - 1.0
else:
    f_height_min = -(f_height - 1.0) / 2.0
    f_height_max = (f_height - 1.0) / 2.0
if f_width % 2 == 0:
    f_width_min = -(f_width / 2.0)
    f_width_max = (f_width / 2.0) - 1.0
else:
    f_width_min = -(f_width - 1.0) / 2.0
    f_width_max = (f_width - 1.0) / 2.0

# 1 x 10 x 10 x 2
# fourier_coord = create_meshgrid(f_width, f_height,
#                                 0, f_width - 1, 0, f_height - 1,
#                                 batch_size=tf.constant(batch_size))
fourier_coord = create_meshgrid(f_width, f_height,
                                f_width_min, f_width_max,
                                f_height_min, f_height_max,
                                batch_size=tf.constant(batch_size))

# cppn_num_neurons x (fourier_basis_size x 2)
cppn_num_neurons = 1
shape = [cppn_num_neurons, fourier_basis_size]
image_basis_real = np.zeros(shape)
# set first x component to 10 (so min rgb val is roughly -1 and max is 1)
image_basis_real[0, 56] = 10.0
image_basis_real = tf.cast(image_basis_real, tf.float32)
image_basis_imag = tf.zeros(shape)
image_basis = tf.concat([image_basis_real, image_basis_imag], -1)

# cppn_num_neurons x fourier_basis_size
complex_image_basis = tf.dtypes.complex(
    image_basis[:, :fourier_basis_size],
    image_basis[:, fourier_basis_size:]
)

# Reshape fourier coord to 1 x fourier_basis_size x 2
f_coord = tf.reshape(fourier_coord, [batch_size, fourier_basis_size, 2])
print(f_coord.eval())

# 1 x 100 x 100 x (cppn_num_neurons x 2)
shape = [batch_size, input_height, input_width, cppn_num_neurons]
dx = tf.ones(shape) * 0.0
dy = tf.ones(shape) * 0.0
shifts = tf.concat([dx, dy], -1)

# (100 x 100) x cppn_num_neurons x (fourier_basis_size x 2)
shifted_image_basis = ApplyShiftsLayer('shift', shifts, complex_image_basis,
                                       fourier_coord)

# 1 x 100 x 100 x cppn_num_neurons
colour_layer_r = tf.ones(shape) * 1
colour_layer_g = tf.ones(shape) * 1
colour_layer_b = tf.ones(shape) * 1

# 1 x 100 x 100 x fourier_basis_size
coeffs_r, coeffs_g, coeffs_b = \
    CombineBasisLayer('combine_basis', colour_layer_r, colour_layer_g,
                      colour_layer_b, shifted_image_basis)

# 1 x 100 x 100 x 1
output_r = IDFTLayer('output_r', input_coord, fourier_coord, coeffs_r)
output_g = IDFTLayer('output_g', input_coord, fourier_coord, coeffs_g)
output_b = IDFTLayer('output_b', input_coord, fourier_coord, coeffs_b)

print('min max output r',
      tf.reduce_min(output_r).eval(),
      tf.reduce_max(output_r).eval())

# 1 x 100 x 100 x 3
output = tf.concat([output_r, output_g, output_b], axis=-1)
output = tf.cast(tf.sigmoid(output) * 255.0, tf.uint8)

result = output.eval()[0]

cv2.imshow('image', result[..., ::-1])
cv2.waitKey()