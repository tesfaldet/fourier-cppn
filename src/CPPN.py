import tensorflow as tf
import numpy as np
import os
from src.layers.ConvLayer import ConvLayer
from src.layers.IDFTLayer import IDFTLayer
from src.utils.create_meshgrid import create_meshgrid
from src.utils.check_snapshots import check_snapshots
from src.PerceptualLoss import PerceptualLoss
from src.utils.load_image import load_image


class CPPN:
    def __init__(self,
                 my_config,
                 tf_config):
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('CPPN Num Variables: ',
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
                init = \
                    tf.initializers \
                      .random_normal(0, tf.sqrt(1.0 / prev_num_channels))
                layer = \
                    ConvLayer(layer_name, prev_layer,
                              out_channels=self.my_config['cppn_num_neurons'],
                              weight_init=init,
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
        
        # Change scaling of fourier meshgrid to match scaling of input meshgrid
        scale_factor = (self.input_width - 1.0) / \
            (self.input_coord_max - self.input_coord_min)
        self.fourier_meshgrid = self.fourier_meshgrid / scale_factor

        # Each output is 1 x H x W x 1
        self.output_r = IDFTLayer('output_r', self.input_meshgrid,
                                  self.fourier_meshgrid, self.coeffs_r)
        self.output_g = IDFTLayer('output_g', self.input_meshgrid,
                                  self.fourier_meshgrid, self.coeffs_g)
        self.output_b = IDFTLayer('output_b', self.input_meshgrid,
                                  self.fourier_meshgrid, self.coeffs_b)

        # Construct RGB output 1 x H x W x 3
        self.output = tf.concat([self.output_r,
                                 self.output_g,
                                 self.output_b], axis=-1)

        self.output = tf.sigmoid(self.output)

        # OBJECTIVE
        target_path = os.path.join(self.my_config['data_dir'],
                                   'textures', 'pebbles.jpg')
        self.target = tf.image.resize_images(
            load_image(target_path), [self.input_width, self.input_height])
        self.loss = 1e5 * \
            PerceptualLoss(self.my_config, self.output,
                           self.target,
                           style_layers=self.my_config['style_layers']
                                            .split(',')).style_loss

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0, tf.uint8))

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)

            # coeffs = (tf.real(self.coeffs_r) +
            #           tf.real(self.coeffs_g) +
            #           tf.real(self.coeffs_b)) / 3.0

            # tf.summary.image('DC', tf.cast(tf.sigmoid(coeffs[..., 0:1])*255.0, tf.uint8))
            # tf.summary.image('horiz', tf.cast(tf.sigmoid(coeffs[..., 5:6])*255.0, tf.uint8))
            # tf.summary.image('vert', tf.cast(tf.sigmoid(coeffs[..., 20:21])*255.0, tf.uint8))

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
                check_snapshots(self.my_config['run_id'])
            writer = tf.summary.FileWriter(
                os.path.join(self.my_config["log_dir"],
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
                               os.path.join(self.my_config['snapshot_dir'],
                                            self.my_config['run_id'],
                                            'snapshot_iter'), global_step=i)

    def validate(self):
        pass

    def predict(self):
        pass
