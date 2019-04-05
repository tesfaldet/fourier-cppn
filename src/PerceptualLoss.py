import tensorflow as tf
import numpy as np
from src.VGG19 import VGG19
from src.layers.MSELayer import MSELayer
from src.utils.vgg import vgg_process


class PerceptualLoss(object):

    # predicted and target are RGB [0, 1]
    def __init__(self, my_config, predicted, target,
                 style_layers=['conv1_1/Relu', 'pool1', 'pool2',
                               'pool3', 'pool4'],
                 content_layers=['conv1_1/Relu']):
        self.my_config = my_config
        self.predicted = predicted
        self.target = target
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.build_graph()
        print('PerceptualLoss Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        with tf.name_scope('PerceptualLoss'):
            # VGG accepts BGR [0-mean, 255-mean] mean subtracted
            vgg19_predicted = VGG19(self.my_config, 'VGG19_predicted',
                                    'PerceptualLoss',
                                    vgg_process(self.predicted))
            vgg19_target = VGG19(self.my_config, 'VGG19_target',
                                 'PerceptualLoss',
                                 vgg_process(self.target))

            self.style_loss = self._style_loss(vgg19_predicted,
                                               vgg19_target,
                                               self.style_layers)
            # self.content_loss = self._content_loss(vgg19_predicted,
            #                                        vgg19_target,
            #                                        self.content_layers)

    def _style_loss(self, vgg_predicted, vgg_target, style_layers):
        with tf.name_scope('StyleLoss'):
            losses = []
            for l in style_layers:
                loss = MSELayer(vgg_predicted.gramian_for_layer(l),
                                vgg_target.gramian_for_layer(l))
                losses.append(loss)
            # return average style loss across layers
            avg_style_loss = tf.add_n(losses) / tf.to_float(len(style_layers))
            return avg_style_loss   

    def _content_loss(self, vgg_predicted, vgg_target, content_layers):
        with tf.name_scope('ContentLoss'):
            losses = []
            for l in content_layers:
                loss = MSELayer(vgg_predicted.activations_for_layer(l),
                                vgg_target.activations_for_layer(l))
                losses.append(loss)
            # return average content loss across layers
            avg_content_loss = \
                tf.add_n(losses) / tf.to_float(len(content_layers))
            return avg_content_loss