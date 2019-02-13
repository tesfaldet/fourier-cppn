import tensorflow as tf
from src.utils.create_meshgrid import create_meshgrid


# transform the image with the given transformation parameters
# input shape [1, height, width, 3]
# transformation shape [3, 3]
def SpatialTransformerLayer(input, transformation, inverse=False):
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


def PhotometricTransformLayer():
    pass
