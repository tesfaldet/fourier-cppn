import os
from cv2 import imwrite


def write_images(images, path, training_iteration=None):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        if training_iteration is None:
            target_path = os.path.join(path, str(i).zfill(8) + '.png')
        else:
            target_path = \
                os.path.join(path, str(training_iteration).zfill(8) + '_' +
                             str(i).zfill(8) + '.png')

        # [0, 1] float32 -> [0, 255] uint8
        im = (images[i] * 255.0).astype('uint8')

        im = im[..., ::-1]  # RGB -> BGR 
        im = imwrite(target_path, im)
