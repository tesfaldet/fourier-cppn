import tensorflow as tf
import numpy as np
from src.utils.create_meshgrid import create_meshgrid
from src.utils.stop_gradients import stop_gradients
from src.layers.ConvLayer import ConvLayer


# transform the image with the given transformation parameters
# input shape [1, height, width, 3]
# transformation shape [3, 3]
def SpatialTransformerLayer(name, input, transformation=None, inverse=False,
                            trainable=True):
    with tf.name_scope(name):
        shape = input.get_shape().as_list()
        height = shape[1]
        width = shape[2]

        # xy_coords shape [1, height, width, 2]
        xy_coords = create_meshgrid(width, height, 0, width-1, 0, height-1)
        x_coords = xy_coords[0, :, :, 0]  # [height, width]
        y_coords = xy_coords[0, :, :, 1]
        z_coords = tf.ones(shape=tf.shape(y_coords))  # homogeneous coords
        xyz_coords = tf.concat([tf.reshape(x_coords, [1, -1]),
                                tf.reshape(y_coords, [1, -1]),
                                tf.reshape(z_coords, [1, -1])], 0)  # [3, N]
        
        if transformation is None:
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                transformation = tf.get_variable('transformation_params',
                                                 initializer=(tf.eye(3) * tf.random_normal([3, 3], 1.0, 0.5)),
                                                 trainable=trainable)
                transformation = stop_gradients(transformation,
                                                np.array([[1, 1, 1],
                                                          [1, 1, 1],
                                                          [0, 0, 0]], dtype=np.float32))

        if inverse is True:
            transformation = tf.matrix_inverse(transformation)
        
        # Apply transformation
        transformed_xyz_coords = tf.matmul(transformation, xyz_coords)  # [3, N]

        warp = tf.reshape(transformed_xyz_coords[:2], [2, height, width])
        warp = tf.transpose(warp, [1, 2, 0])  # [height, width, 2]
        warp = tf.expand_dims(warp, 0)  # [1, height, width, 2]
        # TODO: will need to tile warp to support batch size > 1
        warped = tf.contrib.resampler.resampler(input, warp)

        return warped


def PhotometricTransformLayer(name, input, trainable=True):
    with tf.name_scope(name):
        init = tf.initializers.ones()
        transformed = ConvLayer(name, input, 3,
                                activation='relu',
                                weight_init=init,
                                trainable=trainable)
        return transformed
