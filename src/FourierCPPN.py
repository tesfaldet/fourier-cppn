import tensorflow as tf
import numpy as np
import os

# Import layers
from src.layers.ConvLayer import ConvLayer
from src.layers.IDFTLayer import IDFTLayer

# Import utilities
from src.utils.create_meshgrid import create_meshgrid
from src.utils.create_meshgrid_numpy import create_meshgrid_numpy
from src.utils.check_snapshots import check_snapshots
from src.utils.write_images import write_images

# Some losses
from src.PerceptualLoss import PerceptualLoss


class FourierCPPN:
    def __init__(self,
                 dataset,
                 my_config,
                 tf_config):
        self.dataset = dataset
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('Fourier CPPN Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.global_variables()]))

    def build_graph(self):
        trainable = self.my_config['train']

        # GRAPH DATA
        with tf.name_scope('Data'):
            if trainable:
                self.next_batch = self.dataset.iterator.get_next()
                self.input_coord = self.next_batch[0]  # B x H x W x 2
                self.target = self.next_batch[1]  # B x H x W x 3
                self.index = self.next_batch[2]  # B
            else:
                self.input_coord = \
                    tf.placeholder(tf.float32, shape=[None, None, None, 2])
                self.index = \
                    tf.placeholder(tf.int32, shape=[None])

            self.batch_size = tf.shape(self.input_coord)[0]

            # B x latent size
            self.latent_vector = tf.one_hot(self.index, depth=2)

            # B x 1 x 1 x latent size
            self.latent_vector = \
                tf.reshape(self.latent_vector,
                           [self.batch_size, 1, 1,
                            self.my_config['cppn_latent_size']])
            input_height = tf.shape(self.input_coord)[1]
            input_width = tf.shape(self.input_coord)[2]
            # B x H x W x latent size
            self.latent_vector = tf.tile(self.latent_vector,
                                         [1, input_height, input_width, 1])

            # Input meshgrid in cppn scale B x H x W x 2
            self.input_coord_rescaled = \
                self.input_coord * \
                eval(self.my_config['cppn_coord_scale_factor'])

            self.input_coord_rescaled *= \
                (self.my_config['cppn_latent_size'] / 2.0)

            # B x H x W x (2 + latent size)
            self.input = \
                tf.concat([self.input_coord_rescaled, self.latent_vector],
                          axis=-1)
            self.input.set_shape([None, None, None,
                                  2 + self.my_config['cppn_latent_size']])

        with tf.name_scope('Fourier_Coordinates'):
            self.f_dimensions = \
                self.my_config['cppn_fourier_dimensions'].split(',')
            self.f_width = int(self.f_dimensions[0])
            self.f_height = int(self.f_dimensions[1])
            self.fourier_coord_range = \
                self.my_config['cppn_fourier_coordinate_range'].split(',')
            self.fourier_coord_min = eval(self.fourier_coord_range[0])
            self.fourier_coord_max = eval(self.fourier_coord_range[1])
            self.fourier_basis_size = self.f_width * self.f_height

            # B x H_f x W_f x 2
            self.fourier_coord = \
                create_meshgrid(self.f_width, self.f_height,
                                self.fourier_coord_min, self.fourier_coord_max,
                                self.fourier_coord_min, self.fourier_coord_max,
                                batch_size=self.batch_size)

        # CPPN OUTPUT
        with tf.name_scope('CPPN'):
            self.cppn_layers = [('input', self.input)]
            for i in range(self.my_config['cppn_num_layers']):
                prev_layer = self.cppn_layers[i][1]
                prev_num_channels = tf.cast(prev_layer.shape[-1],
                                            tf.float32)
                layer_name = 'fc' + str(i + 1)
                weight_init = \
                    tf.initializers \
                      .random_normal(0, tf.sqrt(1.0 / prev_num_channels))
                out_channels = self.my_config['cppn_num_neurons']
                activation = self.my_config['cppn_activation']
                if (i + 1) == self.my_config['cppn_num_layers']:
                    out_channels *= 3  # for RGB
                    activation = 'atan'
                layer = \
                    ConvLayer(layer_name, prev_layer,
                              out_channels=out_channels,
                              weight_init=weight_init,
                              activation=activation,
                              trainable=trainable)
                self.cppn_layers.append((layer_name, layer))

            # Split activations into thirds, each corresponding to a colour
            # channel.
            # B x H x W x (cppn_num_neurons x 3) ->
            # B x H x W x cppn_num_neurons
            colour_layer = self.cppn_layers[-1][1] / 0.67
            colour_layer_r = \
                colour_layer[...,
                             :self.my_config['cppn_num_neurons']]
            colour_layer_g = \
                colour_layer[...,
                             self.my_config['cppn_num_neurons']:
                             self.my_config['cppn_num_neurons']*2]
            colour_layer_b = \
                colour_layer[...,
                             self.my_config['cppn_num_neurons']*2:
                             self.my_config['cppn_num_neurons']*3]

            # B x H x W x (fourier_basis_size x 2)
            self.coeffs_r = ConvLayer('coefficients', colour_layer_r,
                                      self.fourier_basis_size * 2,
                                      activation=None)
            self.coeffs_g = ConvLayer('coefficients', colour_layer_g,
                                      self.fourier_basis_size * 2,
                                      activation=None)
            self.coeffs_b = ConvLayer('coefficients', colour_layer_b,
                                      self.fourier_basis_size * 2,
                                      activation=None)
            
            # B x H x W x fourier_basis_size
            self.coeffs_r = tf.dtypes.complex(
                self.coeffs_r[..., :self.fourier_basis_size],
                self.coeffs_r[..., self.fourier_basis_size:])
            self.coeffs_g = tf.dtypes.complex(
                self.coeffs_g[..., :self.fourier_basis_size],
                self.coeffs_g[..., self.fourier_basis_size:])
            self.coeffs_b = tf.dtypes.complex(
                self.coeffs_b[..., :self.fourier_basis_size],
                self.coeffs_b[..., self.fourier_basis_size:])

        with tf.name_scope('IDFT'):
            # Each output is B x H x W x 1
            self.output_r = IDFTLayer('output_r', self.input_coord,
                                      self.fourier_coord, self.coeffs_r)
            self.output_g = IDFTLayer('output_g', self.input_coord,
                                      self.fourier_coord, self.coeffs_g)
            self.output_b = IDFTLayer('output_b', self.input_coord,
                                      self.fourier_coord, self.coeffs_b)

        with tf.name_scope('Output'):
            # Construct RGB output B x H x W x 3
            self.output = tf.sigmoid(tf.concat([self.output_r,
                                                self.output_g,
                                                self.output_b], axis=-1))

        # OBJECTIVE
        if self.my_config['train']:
            self.loss = 1e5 * \
                PerceptualLoss(self.my_config, self.output, self.target,
                               style_layers=self.my_config['style_layers']
                                                .split(',')).content_loss

            # Average loss over batch
            self.loss = self.loss / tf.cast(self.batch_size, tf.float32)

            self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0, tf.uint8),
                             max_outputs=2)
            tf.summary.image('Target', tf.cast(self.target * 255.0, tf.uint8),
                             max_outputs=2)

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)

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

        saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)

        saved_iterator = \
            tf.data.experimental\
              .make_saveable_from_iterator(self.dataset.iterator)
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saved_iterator)

        with tf.Session(config=self.tf_config) as sess:
            resume, self.iterations_so_far = \
                check_snapshots(self.my_config['run_id'],
                                self.my_config['force_train_from_scratch'])
            self.writer = tf.summary.FileWriter(
                os.path.join(self.my_config['log_dir'],
                             self.my_config['run_id']), sess.graph)

            if resume:
                saver.restore(sess, resume)
            else:
                sess.run(tf.global_variables_initializer())

            train_handle = sess.run(self.dataset.get_training_handle())

            if self.my_config['use_bfgs']:
                train_feed_dict = {global_step: self.iterations_so_far,
                                   self.dataset.handle: train_handle}

                opt.minimize(sess,
                             fetches=[self.loss, self.summaries, self.output],
                             feed_dict=train_feed_dict,
                             loss_callback=self.minimize_callback)

                # Snapshot at end of optimization since BFGS updates vars
                # at the end
                print('Saving Snapshot...')
                saver.save(sess,
                           os.path.join(self.my_config['snap_dir'],
                                        self.my_config['run_id'],
                                        'snapshot_iter'),
                           global_step=self.iterations_so_far)
            else:
                for i in range(self.iterations_so_far,
                               self.my_config['iterations']):
                    train_feed_dict = {global_step: self.iterations_so_far,
                                       self.dataset.handle: train_handle}
                    results = sess.run([train_step, self.loss,
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
                        saver.save(sess,
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

    def predict(self, model_path):
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(model_path)

        with tf.Session(config=self.tf_config) as sess:
            saver.restore(sess, checkpoint_path)
        
            input_coord = \
                create_meshgrid_numpy(1000, 1000, -200, 200, -200, 200)
            
            index = np.array([0], dtype='int32')
            feed_dict = {self.input_coord: input_coord,
                         self.index: index}

            output = sess.run(self.output, feed_dict=feed_dict)
            target_path = os.path.join(self.my_config['data_dir'],
                                       'out', 'predict')
            write_images(output, target_path)
