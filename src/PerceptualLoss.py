import tensorflow as tf
import numpy as np
from src.VGG19 import VGG19
from src.layers.MSELayer import MSELayer
from src.utils.vgg import vgg_process


class PerceptualLoss(object):

    # predicted and target are RGB [0, 1]
    def __init__(self, name, my_config, predicted, target,
                 style_layers=['conv1_1/Relu', 'pool1', 'pool2',
                               'pool3', 'pool4'],
                 content_layers=['conv4_2/Relu']):
        self.name = name
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
        with tf.name_scope(self.name):
            # VGG accepts BGR [0-mean, 255-mean] mean subtracted
            vgg19_predicted = VGG19(self.my_config, 'VGG19_predicted',
                                    self.name, vgg_process(self.predicted))
            vgg19_target = VGG19(self.my_config, 'VGG19_target',
                                 self.name, vgg_process(self.target))

            self.avg_style_loss, self.style_losses = \
                self._style_loss(vgg19_predicted,
                                 vgg19_target,
                                 self.style_layers)
            self.avg_content_loss, self.content_losses = \
                self._content_loss(vgg19_predicted,
                                   vgg19_target,
                                   self.content_layers)

    def _style_loss(self, vgg_predicted, vgg_target, style_layers):
        with tf.name_scope('StyleLoss'):
            losses = {}
            for l in style_layers:
                loss = MSELayer(vgg_predicted.gramian_for_layer(l),
                                vgg_target.gramian_for_layer(l))
                losses[l] = loss
            # return average style loss across layers
            avg_style_loss = \
                tf.add_n(list(losses.values())) / \
                tf.to_float(len(style_layers))
            return avg_style_loss, losses

    def _content_loss(self, vgg_predicted, vgg_target, content_layers):
        with tf.name_scope('ContentLoss'):
            losses = {}
            for l in content_layers:
                loss = MSELayer(vgg_predicted.activations_for_layer(l),
                                vgg_target.activations_for_layer(l))
                losses[l] = loss
            # return average content loss across layers
            avg_content_loss = \
                tf.add_n(list(losses.values())) / \
                tf.to_float(len(content_layers))
            return avg_content_loss, losses
