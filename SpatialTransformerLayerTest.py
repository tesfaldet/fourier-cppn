import tensorflow as tf
import numpy as np
import cv2
from src.utils.load_image import load_image
import os
from src.layers.TransformLayers import SpatialTransformerLayer

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
im = load_image('data/cppn_sunflowers.png')[..., ::-1]

# add batch dimension
im = np.expand_dims(im, 0)

im_shape = im.shape[1:]

# affine transformation
transformation = tf.constant([[1, 2, 50],
                              [0, 1, 25],
                              [0, 0, 1]], dtype=tf.float32)

# preparing placeholders
input = tf.placeholder(dtype=tf.float32, shape=[None] + list(im_shape),
                       name='input')

# main computation
with tf.device('/gpu:' + gpu):
    warped = SpatialTransformerLayer(input, transformation)


# retrieve output
with tf.Session(config=config_proto) as sess:
    out = sess.run(warped, feed_dict={input: im})

# show output
cv2.imshow('im1', im[0])
cv2.imshow('im2', out[0])
cv2.waitKey()