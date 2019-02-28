import tensorflow as tf
import numpy as np
from src.layers.TransformLayers import \
    SpatialTransformerLayer, PhotometricTransformLayer


# input needs to be between [0, 1]
class TextureModule:
    def __init__(self,
                 name,
                 input):
        self.name = name
        self.texture = input
        self.build_graph()
        print(name + 'Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        with tf.name_scope(self.name):
            geometrically_transformed = \
                SpatialTransformerLayer('SpatialTransform_' + self.name,
                                        self.texture, trainable=False)

            photometrically_transformed = \
                PhotometricTransformLayer('PhotoTransform_' + self.name,
                                          geometrically_transformed)

            # output is between [0, 1]
            self.output = photometrically_transformed

    def build_summaries(self):
        pass