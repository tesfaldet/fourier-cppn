import os
from src.layers.GramLayer import GramLayer
import tensorflow as tf


class VGG19(object):

    def __init__(self, my_config, name, prefix, input):
        self.my_config = my_config
        self.name = name
        self.full_name = prefix + '/' + self.name

        vgg_path = os.path.join(self.my_config['data_dir'], 'models',
                                'vgg19_normalized_valid_V1proto.tfmodel')
        with open(vgg_path, mode='rb') as f:
            file_content = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_content)

        # 'images' is the name of the input for the imported vgg graph
        tf.import_graph_def(graph_def, input_map={'images': input},
                            name=self.name)

    def gramian_for_layer(self, layer):
        activations = self.activations_for_layer(layer)

        # Reshape from (batch, height, width, channels) to
        # (batch, channels, height, width)
        shuffled_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        return GramLayer(shuffled_activations, normalize_method='ulyanov')

    def activations_for_layer(self, layer):
        return tf.get_default_graph() \
                 .get_tensor_by_name('{0}/{1}:0'
                                     .format(self.full_name, layer))
