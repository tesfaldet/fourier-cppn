import tensorflow as tf
import numpy as np
import os
from src.layers.ConvLayer import ConvLayer
from src.utils.create_meshgrid import create_meshgrid
from src.utils.check_snapshots import check_snapshots
from src.PerceptualLoss import PerceptualLoss
from src.utils.load_image import load_image
from src.utils.write_images import write_images
from src.InceptionV1 import InceptionV1


class RGBCPPN:
    def __init__(self,
                 my_config,
                 tf_config):
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('RGB CPPN Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.global_variables()]))

    def build_graph(self):
        # COORDINATE MESHGRID INPUT
        with tf.name_scope('Input_Meshgrid'):
            # TODO: clean this up
            self.input_dimensions = \
                self.my_config['input_dimensions'].split(',')
            self.input_width = int(self.input_dimensions[0])
            self.input_height = int(self.input_dimensions[1])
            self.input_coord_range = \
                self.my_config['cppn_input_coordinate_range'].split(',')
            self.input_coord_min = eval(self.input_coord_range[0])
            self.input_coord_max = eval(self.input_coord_range[1])
            self.input_meshgrid = \
                create_meshgrid(self.input_width, self.input_height,
                                self.input_coord_min, self.input_coord_max,
                                self.input_coord_min, self.input_coord_max)

        # CPPN OUTPUT
        with tf.name_scope('CPPN'):
            self.cppn_layers = [('input', self.input_meshgrid)]
            for i in range(self.my_config['cppn_num_layers']):
                prev_layer = self.cppn_layers[i][1]
                prev_num_channels = tf.cast(tf.shape(prev_layer)[-1],
                                            tf.float32)
                layer_name = 'fc' + str(i + 1)
                weight_init = \
                    tf.initializers \
                      .random_normal(0, tf.sqrt(1.0 / prev_num_channels))
                layer = \
                    ConvLayer(layer_name, prev_layer,
                              out_channels=self.my_config['cppn_num_neurons'],
                              weight_init=weight_init,
                              activation=self.my_config['cppn_activation'])
                self.cppn_layers.append((layer_name, layer))

            # Outputting RGB
            self.output = ConvLayer('rgb', self.cppn_layers[-1][1],
                                    3, activation='sigmoid')

        # OBJECTIVE
        target_path = os.path.join(self.my_config['data_dir'],
                                   'textures',
                                   self.my_config['target_image_name'])
        self.target_dimensions = self.my_config['target_dimensions'].split(',')
        self.target_width = int(self.target_dimensions[0])
        self.target_height = int(self.target_dimensions[1])
        self.target = tf.image.resize_images(
            load_image(target_path), [self.target_width, self.target_height])
        self.loss = 1e5 * \
            PerceptualLoss(self.my_config, self.output,
                           self.target,
                           style_layers=self.my_config['style_layers']
                                            .split(',')).style_loss
        # self.loss = -InceptionV1('InceptionV1Loss', self.output)\
        #     .avg_channel("mixed4b_3x3_pre_relu", 77)
        # self.loss = -InceptionV1('InceptionV1Loss', self.output)\
        #     .avg_channel('mixed4b_pool_reduce_pre_relu', 16)

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0, tf.uint8))

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)

            tf.summary.scalar('RGB_min', tf.reduce_min(self.output))
            tf.summary.scalar('RGB_max', tf.reduce_max(self.output))
            tf.summary.scalar('RGB_mean', tf.reduce_mean(self.output))

            # Merge all summaries
            self.summaries = tf.summary.merge_all()

    def train(self):
        global_step = tf.Variable(0, trainable=False)

        opt = tf.train.AdamOptimizer(
            learning_rate=self.my_config['learning_rate'])
        train_step = opt.minimize(self.loss)

        saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)

        with tf.Session(config=self.tf_config) as sess:
            resume, iterations_so_far = \
                check_snapshots(self.my_config['run_id'],
                                self.my_config['force_train_from_scratch'])
            writer = tf.summary.FileWriter(
                os.path.join(self.my_config['log_dir'],
                             self.my_config['run_id']), sess.graph)

            if resume:
                saver.restore(sess, resume)
            else:
                sess.run(tf.global_variables_initializer())

            for i in range(iterations_so_far, self.my_config['iterations']):
                train_feed_dict = {global_step: i}
                results = sess.run([train_step, self.loss, self.summaries],
                                   feed_dict=train_feed_dict)
                loss = results[1]
                train_summary = results[2]

                # Saving/Logging
                if i % self.my_config['print_frequency'] == 0:
                    print('(' + self.my_config['run_id'] + ') ' +
                          'Iteration ' + str(i) +
                          ', Loss: ' + str(loss))

                if i % self.my_config['log_frequency'] == 0:
                    writer.add_summary(train_summary, i)
                    writer.flush()
                
                if i % self.my_config['snapshot_frequency'] == 0 and \
                   i != iterations_so_far:
                    print('Saving Snapshot...')
                    saver.save(sess,
                               os.path.join(self.my_config['snap_dir'],
                                            self.my_config['run_id'],
                                            'snapshot_iter'), global_step=i)

    def validate(self):
        pass

    def predict(self, model_path):
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(model_path)

        with tf.Session(config=self.tf_config) as sess:
            saver.restore(sess, checkpoint_path)
            output = sess.run(self.output)
            target_path = os.path.join(self.my_config['data_dir'], 'out',
                                       'predict')
            write_images(output, target_path)
