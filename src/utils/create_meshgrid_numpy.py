import numpy as np


def create_meshgrid_numpy(width, height, minval_x=-1.,
                          maxval_x=1., minval_y=-1., maxval_y=1.,
                          batch_size=1):
    x_coords, y_coords = \
        np.meshgrid(np.linspace(minval_x, maxval_x, width),
                    np.linspace(minval_y, maxval_y, height))
    xy_coords = np.stack([x_coords, y_coords], 2)  # [height, width, 2]
    if batch_size > 0:
        xy_coords = np.expand_dims(xy_coords, axis=0)  # [1, height, width, 2]
    if batch_size > 1:
        xy_coords = np.tile(xy_coords, [batch_size, 1, 1, 1])
    return xy_coords.astype('float32')  # +x right, +y down