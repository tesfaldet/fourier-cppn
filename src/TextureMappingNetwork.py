import tensorflow as tf
import numpy as np
from src.TextureModule import TextureModule
from src.utils.load_image import load_image


class TextureMappingNetwork:
    def __init__(self):
        self.build_graph()
        print('TextureMappingNetwork Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        with tf.name_scope('TextureMappingNetwork'):
            # 224x224 RGB basis textures in range [0, 1]
            basis_1 = tf.image.resize_images(
                load_image('data/textures/1.1.01.tiff'), [224, 224])

            # Texture modules collectively forming the basis set
            module_1 = TextureModule('module_1', basis_1)
            module_2 = TextureModule('module_2', basis_1)
            module_3 = TextureModule('module_3', basis_1)
            self.output = module_1.output + module_2.output + module_3.output
            # self.output = module_1.output

    def build_summaries(self):
        pass
