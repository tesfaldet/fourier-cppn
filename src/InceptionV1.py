import tensorflow as tf
from lucid.modelzoo import vision_models


class InceptionV1(object):

    def __init__(self, name, input):
        self.name = name

        self.model = vision_models.InceptionV1()

        # Load GraphDef to model.graph_def
        self.model.load_graphdef()

        # Import GraphDef to current graph with given scope and tensor input
        self.model.import_graph(t_input=input, scope=self.name)

        # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    def get_layer(self, name):
        return tf.get_default_graph()\
            .get_tensor_by_name('{0}/{1}:0'.format(self.name, name))\

    # reduce mean spatially for a given channel of a layer
    def avg_channel(self, name, n_channel):
        layer = self.get_layer(name)
        return tf.reduce_mean(layer[..., n_channel])