import tensorflow as tf
import numpy as np
from src.TextureModule import TextureModule
from src.utils.load_image import load_image
from src.utils.create_image import cos_pattern_horizontal, cos_pattern_vertical


class TextureMappingNetwork:
    def __init__(self):
        self.build_graph()
        print('TextureMappingNetwork Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        with tf.name_scope('TextureMappingNetwork'):
            # 224x224 RGB basis textures in range [0, 1]
            basis_1 = tf.to_float(cos_pattern_horizontal(224, 15))
            basis_2 = tf.to_float(cos_pattern_vertical(224, 15))
            basis_3 = tf.image.resize_images(
                load_image('data/cat.jpg'), [224, 224])

            # Texture modules collectively forming the basis set
            module_1 = TextureModule('module_1', basis_1)
            module_2 = TextureModule('module_2', basis_2)
            # module_3 = TextureModule('module_3', basis_3)
            self.output = tf.add_n([module_1.output,
                                    module_2.output]) * 0.5