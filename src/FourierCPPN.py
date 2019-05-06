import tensorflow as tf
import numpy as np
import os
from src.layers.ConvLayer import ConvLayer
from src.layers.IDFTLayer import IDFTLayer
from src.utils.create_meshgrid import create_meshgrid
from src.utils.check_snapshots import check_snapshots
from src.PerceptualLoss import PerceptualLoss
from src.utils.load_image import load_image
from src.utils.write_images import write_images
from src.InceptionV1 import InceptionV1
from src.layers.MSELayer import MSELayer


class FourierCPPN:
    def __init__(self,
                 my_config,
                 tf_config):
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('Fourier CPPN Num Variables: ',
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
                                self.input_coord_min, self.input_coord_max,
                                2)
            class_1 = tf.tile(tf.reshape(np.array([0., 1.],
                                                  dtype=np.float32),
                                         [1, 1, 1, 2]),
                              [1, self.input_height, self.input_width, 1])
            class_2 = tf.tile(tf.reshape(np.array([1., 0.],
                                                  dtype=np.float32),
                                         [1, 1, 1, 2]),
                              [1, self.input_height, self.input_width, 1])
            classes = tf.concat([class_1, class_2], axis=0)
            self.input_meshgrid = tf.concat([self.input_meshgrid, classes],
                                            axis=-1)

        with tf.name_scope('Fourier_Meshgrid'):
            self.f_dimensions = \
                self.my_config['cppn_fourier_dimensions'].split(',')
            self.f_width = int(self.f_dimensions[0])
            self.f_height = int(self.f_dimensions[1])
            self.fourier_coord_range = \
                self.my_config['cppn_fourier_coordinate_range'].split(',')
            self.fourier_coord_min = eval(self.fourier_coord_range[0])
            self.fourier_coord_max = eval(self.fourier_coord_range[1])
            self.fourier_basis_size = self.f_width * self.f_height
            self.fourier_meshgrid = \
                create_meshgrid(self.f_width, self.f_height,
                                self.fourier_coord_min, self.fourier_coord_max,
                                self.fourier_coord_min, self.fourier_coord_max)

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

            # Outputting amplitudes aka Fourier mixin coefficients for a
            # basis set of exponentials aka sinusoids at different frequencies
            # 1 x H x W x (H_f x W_f x 2)
            self.coeffs = ConvLayer('coefficients', self.cppn_layers[-1][1],
                                    self.fourier_basis_size * 2 * 3,
                                    activation=None)

            # Make fourier coefficients complex 1 x H x W x (H_f x W_f)
            self.coeffs_r = tf.dtypes.complex(
                 self.coeffs[..., :self.fourier_basis_size*1],
                 self.coeffs[...,
                             self.fourier_basis_size*1:
                             self.fourier_basis_size*2])
            self.coeffs_g = tf.dtypes.complex(
                 self.coeffs[...,
                             self.fourier_basis_size*2:
                             self.fourier_basis_size*3],
                 self.coeffs[...,
                             self.fourier_basis_size*3:
                             self.fourier_basis_size*4])
            self.coeffs_b = tf.dtypes.complex(
                 self.coeffs[...,
                             self.fourier_basis_size*4:
                             self.fourier_basis_size*5],
                 self.coeffs[...,
                             self.fourier_basis_size*5:
                             self.fourier_basis_size*6])

        # Input meshgrid in pixel scale
        self.input_meshgrid_rescaled = \
            create_meshgrid(self.input_width, self.input_height,
                            0, self.input_width - 1,
                            0, self.input_height - 1)

        # Each output is 1 x H x W x 1
        self.output_r = IDFTLayer('output_r', self.input_meshgrid_rescaled,
                                  self.fourier_meshgrid, self.coeffs_r)
        self.output_g = IDFTLayer('output_g', self.input_meshgrid_rescaled,
                                  self.fourier_meshgrid, self.coeffs_g)
        self.output_b = IDFTLayer('output_b', self.input_meshgrid_rescaled,
                                  self.fourier_meshgrid, self.coeffs_b)

        # Construct RGB output 1 x H x W x 3
        self.output_pre_sigmoid = tf.concat([self.output_r,
                                             self.output_g,
                                             self.output_b], axis=-1)

        self.output = tf.sigmoid(self.output_pre_sigmoid)

        # OBJECTIVE
        target_path_1 = os.path.join(self.my_config['data_dir'],
                                     'textures',
                                     'peppers.jpg')
        target_path_2 = os.path.join(self.my_config['data_dir'],
                                     'textures',
                                     'pebbles.jpg')
        self.target_dimensions = self.my_config['target_dimensions'].split(',')
        self.target_width = int(self.target_dimensions[0])
        self.target_height = int(self.target_dimensions[1])
        self.target_1 = tf.image.resize_images(
            load_image(target_path_1), [self.target_height, self.target_width])
        self.target_2 = tf.image.resize_images(
            load_image(target_path_2), [self.target_height, self.target_width])
        self.target = tf.concat([self.target_1, self.target_2], axis=0)
        self.loss = 1e5 * \
            PerceptualLoss(self.my_config, self.output, self.target,
                           style_layers=self.my_config['style_layers']
                                            .split(',')).style_loss
        # self.loss = -InceptionV1('InceptionV1Loss', self.output)\
        #     .avg_channel("mixed4b_3x3_pre_relu", 77)
        # self.loss = -InceptionV1('InceptionV1Loss', self.output)\
        #     .avg_channel('mixed4b_pool_reduce_pre_relu', 16)
        # target_path = os.path.join(self.my_config['data_dir'],
        #                            'mattie.jpg')
        # self.target = tf.image.resize_images(
        #     load_image(target_path), [self.target_height, self.target_width])
        # self.loss = 1e5 * MSELayer(self.output, self.target)
        # self.loss += 1e5 * \
        #     PerceptualLoss(self.my_config, self.output,
        #                    self.target,
        #                    style_layers=self.my_config['content_layers']
        #                                     .split(',')).content_loss

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0, tf.uint8))
            tf.summary.image('Target', self.target)

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)

            tf.summary.scalar('RGB_min', tf.reduce_min(self.output))
            tf.summary.scalar('RGB_max', tf.reduce_max(self.output))
            tf.summary.scalar('RGB_mean', tf.reduce_mean(self.output))

            tf.summary.scalar('RGB_pre_sigmoid_min',
                              tf.reduce_min(self.output_pre_sigmoid))
            tf.summary.scalar('RGB_pre_sigmoid_max',
                              tf.reduce_max(self.output_pre_sigmoid))
            tf.summary.scalar('RGB_pre_sigmoid_mean',
                              tf.reduce_mean(self.output_pre_sigmoid))

            # Merge all summaries
            self.summaries = tf.summary.merge_all()

    def train(self):
        global_step = tf.Variable(0, trainable=False)

        if self.my_config['use_bfgs']:
            opt = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss, method='L-BFGS-B',
                options={'maxfun': self.my_config['iterations']})
        else:
            opt = tf.train.AdamOptimizer(
                learning_rate=self.my_config['learning_rate'])
            train_step = opt.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)

        with tf.Session(config=self.tf_config) as self.sess:
            resume, self.iterations_so_far = \
                check_snapshots(self.my_config['run_id'],
                                self.my_config['force_train_from_scratch'])
            self.writer = tf.summary.FileWriter(
                os.path.join(self.my_config['log_dir'],
                             self.my_config['run_id']), self.sess.graph)

            if resume:
                self.saver.restore(self.sess, resume)
            else:
                self.sess.run(tf.global_variables_initializer())

            if self.my_config['use_bfgs']:
                opt.minimize(self.sess,
                             fetches=[self.loss, self.summaries, self.output],
                             feed_dict={global_step: self.iterations_so_far},
                             loss_callback=self.minimize_callback)

                # Snapshot at end of optimization since BFGS updates vars
                # at end
                print('Saving Snapshot...')
                self.saver.save(self.sess,
                                os.path.join(self.my_config['snap_dir'],
                                             self.my_config['run_id'],
                                             'snapshot_iter'),
                                global_step=self.iterations_so_far)
            else:
                for i in range(self.iterations_so_far,
                               self.my_config['iterations']):
                    train_feed_dict = {global_step: i}
                    results = self.sess.run([train_step, self.loss,
                                             self.summaries, self.output],
                                            feed_dict=train_feed_dict)
                    loss = results[1]
                    train_summary = results[2]
                    output = results[3]

                    # Saving/Logging
                    if i % self.my_config['print_frequency'] == 0:
                        print('(' + self.my_config['run_id'] + ') ' +
                              'Iteration ' + str(i) +
                              ', Loss: ' + str(loss))

                    if i % self.my_config['log_frequency'] == 0:
                        self.writer.add_summary(train_summary, i)
                        self.writer.flush()

                    if i % self.my_config['snapshot_frequency'] == 0 and \
                       i != self.iterations_so_far:
                        print('Saving Snapshot...')
                        self.saver.save(self.sess,
                                        os.path.join(self.my_config['snap_dir'],
                                                     self.my_config['run_id'],
                                                     'snapshot_iter'),
                                        global_step=i)
                    
                    if i % self.my_config['write_frequency'] == 0:
                        target_path = os.path.join(self.my_config['data_dir'],
                                                   'out', 'train',
                                                   self.my_config['run_id'])
                        write_images(output, target_path, training_iteration=i)

    def minimize_callback(self, loss, summaries, output):
        i = self.iterations_so_far

        # Saving/Logging
        if i % self.my_config['print_frequency'] == 0:
            print('(' + self.my_config['run_id'] + ') ' +
                  'Iteration ' + str(i) +
                  ', Loss: ' + str(loss))

        if i % self.my_config['log_frequency'] == 0:
            self.writer.add_summary(summaries, i)
            self.writer.flush()

        if i % self.my_config['write_frequency'] == 0:
            target_path = os.path.join(self.my_config['data_dir'], 'out',
                                       'train', self.my_config['run_id'])
            write_images(output, target_path, training_iteration=i)

        self.iterations_so_far += 1

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
