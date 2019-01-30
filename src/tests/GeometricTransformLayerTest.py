import tensorflow as tf
import numpy as np
import cv2
import src.utils.load_image as load_image
import os
from src.layers.TransformLayers import GeometricTransformLayer

# prevent tf from using other GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# configs
gpu = '0'
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False

# loading data
im = load_image('data/textures/1.1.01.tiff')

# add batch dimension
im = np.expand_dims(im, 0)

im_shape = im.shape[1:]

# preparing placeholders
input = tf.placeholder(dtype=tf.float32, shape=[None] + list(im_shape),
                       name='input')

# transform params
slant = np.radians(22.5)
tilt = np.radians(0)
f = -400
z_0 = -100


# main computation
with tf.device('/gpu:' + gpu):
    warped = GeometricTransformLayer(input, slant, tilt, f, z_0)


# retrieve output
with tf.Session(config=config_proto) as sess:
    out = sess.run(warped, feed_dict={input: im})

# show output
cv2.imshow('im1', im[0])
cv2.imshow('im2', out[0])
cv2.waitKey()