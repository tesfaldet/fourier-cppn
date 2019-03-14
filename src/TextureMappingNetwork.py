import os
import tensorflow as tf
import numpy as np
from src.TextureModule import TextureModule
from src.utils.load_image import load_image


class TextureMappingNetwork:
    def __init__(self, my_config):
        self.my_config = my_config
        self.build_graph()
        print('TextureMappingNetwork Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.global_variables()]))

    def build_graph(self):
        with tf.name_scope('TextureMappingNetwork'):
            # 224x224 RGB basis textures in range [0, 1]
            basis_1_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.01.tiff')
            basis_2_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.02.tiff')
            basis_3_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.03.tiff')
            basis_4_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.04.tiff')
            basis_5_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.05.tiff')
            basis_6_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.06.tiff')
            basis_7_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.07.tiff')
            basis_8_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.08.tiff')
            basis_9_path = os.path.join(self.my_config['data_dir'],
                                        'textures', '1.1.09.tiff')
            basis_10_path = os.path.join(self.my_config['data_dir'],
                                         'textures', '1.1.10.tiff')
            basis_11_path = os.path.join(self.my_config['data_dir'],
                                         'textures', '1.1.11.tiff')
            basis_12_path = os.path.join(self.my_config['data_dir'],
                                         'textures', '1.1.12.tiff')
            basis_13_path = os.path.join(self.my_config['data_dir'],
                                         'textures', '1.1.13.tiff')
            # TODO: clean this up
            dimensions = self.my_config['dimensions'].split(',')
            width = int(dimensions[0])
            height = int(dimensions[1])
            basis_1 = tf.image.resize_images(
                load_image(basis_1_path), [width, height])
            basis_2 = tf.image.resize_images(
                load_image(basis_2_path), [width, height])
            basis_3 = tf.image.resize_images(
                load_image(basis_3_path), [width, height])
            basis_4 = tf.image.resize_images(
                load_image(basis_4_path), [width, height])
            basis_5 = tf.image.resize_images(
                load_image(basis_5_path), [width, height])
            basis_6 = tf.image.resize_images(
                load_image(basis_6_path), [width, height])
            basis_7 = tf.image.resize_images(
                load_image(basis_7_path), [width, height])
            basis_8 = tf.image.resize_images(
                load_image(basis_8_path), [width, height])
            basis_9 = tf.image.resize_images(
                load_image(basis_9_path), [width, height])
            basis_10 = tf.image.resize_images(
                load_image(basis_10_path), [width, height])
            basis_11 = tf.image.resize_images(
                load_image(basis_11_path), [width, height])
            basis_12 = tf.image.resize_images(
                load_image(basis_12_path), [width, height])
            basis_13 = tf.image.resize_images(
                load_image(basis_13_path), [width, height])

            # Texture modules collectively forming the basis set
            module_1 = TextureModule('module_1', basis_1)
            module_2 = TextureModule('module_2', basis_2)
            module_3 = TextureModule('module_3', basis_3)
            module_4 = TextureModule('module_4', basis_4)
            module_5 = TextureModule('module_5', basis_5)
            module_6 = TextureModule('module_6', basis_6)
            module_7 = TextureModule('module_7', basis_7)
            module_8 = TextureModule('module_8', basis_8)
            module_9 = TextureModule('module_9', basis_9)
            module_10 = TextureModule('module_10', basis_10)
            module_11 = TextureModule('module_11', basis_11)
            module_12 = TextureModule('module_12', basis_12)
            module_13 = TextureModule('module_13', basis_13)
            self.output = tf.nn.tanh(tf.add_n([module_1.output,
                                               module_2.output,
                                               module_3.output,
                                               module_4.output,
                                               module_5.output,
                                               module_6.output,
                                               module_7.output,
                                               module_8.output,
                                               module_9.output,
                                               module_10.output,
                                               module_11.output,
                                               module_12.output,
                                               module_13.output]))