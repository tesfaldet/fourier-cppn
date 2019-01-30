import tensorflow as tf
import numpy as np


# warp the image to the given the slant, tilt, and surface distance.
# focal length needed
def GeometricTransformLayer(input, slant, tilt, f, z_0):
	im_shape = input.get_shape().as_list()[1:]

	# meshgrid for warping (in image coords, y down)
	# (0,0) is the centre of the image
	x_coords, y_coords = tf.meshgrid(range(-im_shape[1]/2, im_shape[1]/2),
		                             range(im_shape[0]/2, -im_shape[0]/2, -1))

	# reshape, concat, and cast to float32
	xy_coords = tf.cast(tf.concat([tf.reshape(x_coords, [1, -1]),
		                           tf.reshape(y_coords, [1, -1])], 0), tf.float32)  # [2, N]

	# image-to-surface (inverse of s2i)
	dot = tf.matmul(tf.reshape([np.cos(tilt), np.sin(tilt)], [1, 2]), xy_coords)  # [1, N]
	z = z_0 / (np.sin(slant)*dot + f*np.cos(slant))  # [1, N]
	i2s = tf.reshape([[np.cos(tilt), np.sin(tilt)],
	                  [-np.cos(slant)*np.sin(tilt), np.cos(slant)*np.cos(tilt)]],
	                 [2, 2])  # [2, 2]
	xy_coords = z * tf.matmul(i2s, xy_coords)  # [2, N]
	xy_coords = tf.stack([xy_coords[0]+im_shape[1]/2, -(xy_coords[1]-im_shape[0]/2)], 0)  # switching coord systems
	warp = tf.reshape(xy_coords, [2, im_shape[0], im_shape[1]])
	warp = tf.transpose(warp, [1, 2, 0])
	warp = tf.expand_dims(warp, 0)
	warped = tf.contrib.resampler.resampler(input, warp)

	return warped


def PhotometricTransformLayer():
    pass
