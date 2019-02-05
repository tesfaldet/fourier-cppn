from cv2 import imread


def load_image(path):
    im = imread(path).astype('float32')  # [0, 255]
    im = im[..., ::-1]  # BGR -> RGB
    return im / 255.0  # [0, 1]
