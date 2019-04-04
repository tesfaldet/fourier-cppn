import tensorflow as tf
import numpy as np
import cv2
from src.layers.IDFTLayer import IDFTLayer
from src.utils.create_meshgrid import create_meshgrid


# configs
gpu = '0'
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False

input_meshgrid = create_meshgrid(100, 100, 0, 99, 0, 99)
fourier_meshgrid = create_meshgrid(10, 10, 0, 9, 0, 9)
real = np.zeros([1, 100, 100, 10*10], dtype=np.float32)
real[..., 11] = 100.0  # DC component
imag = tf.zeros_like(real)
coefficients = tf.dtypes.complex(real, imag)
output = IDFTLayer('IDFT', input_meshgrid, fourier_meshgrid, coefficients)
output = tf.sigmoid(output)

with tf.Session() as sess:
    out = sess.run(output)

cv2.imshow('im', out[0])
cv2.waitKey()
cv2.destroyAllWindows()