import tensorflow as tf
import numpy as np
from src.layers import *


class TextureMappingNetwork:
    def __init__(self,
                 config):
        self.config = config
        self.build_graph()
        print('Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        pass

    def build_summaries(self):
        pass
