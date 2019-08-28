import tensorflow as tf
import numpy as np
import os

# Import layers
from src.layers.ConvLayer import ConvLayer

# Import utilities
from src.utils.create_meshgrid_numpy import create_meshgrid_numpy
from src.utils.check_snapshots import check_snapshots
from src.utils.write_images import write_images

# Some losses
from src.PerceptualLoss import PerceptualLoss


class CPPN:
    def __init__(self,
                 dataset,
                 my_config,
                 tf_config):
        self.dataset = dataset
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('CPPN Num Variables: ',
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
                # B x latent size
                self.latent_vector_feed = tf.one_hot(self.index, depth=2)
                # self.latent_vector_feed = \
                #     EmbeddingLayer('Embeddings', self.index,
                #                    self.my_config['cppn_latent_size'], 38)
            else:
                self.input_coord = \
                    tf.placeholder(tf.float32, shape=[None, None, None, 2])
                self.latent_vector_feed = \
                    tf.placeholder(tf.float32, shape=[None, None])

            self.batch_size = tf.shape(self.input_coord)[0]

            # B x 1 x 1 x latent size
            self.latent_vector = \
                tf.reshape(self.latent_vector_feed,
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
                layer = \
                    ConvLayer(layer_name, prev_layer,
                              out_channels=out_channels,
                              weight_init=weight_init,
                              activation=activation,
                              trainable=trainable)
                self.cppn_layers.append((layer_name, layer))

        with tf.name_scope('Output'):
            # Construct RGB output B x H x W x 3
            self.output = ConvLayer('rgb', self.cppn_layers[-1][1],
                                    3, activation='sigmoid')

        # OBJECTIVE
        if self.my_config['train']:
            # self.style_loss = \
            #     PerceptualLoss('PerceptualLoss_style',
            #                    self.my_config, self.output, self.target,
            #                    style_layers=self.my_config['style_layers']
            #                                     .split(','))
            # self.avg_style_loss = 1e9 * self.style_loss.avg_style_loss
            # self.style_losses = self.style_loss.style_losses
            self.avg_style_loss = tf.constant(0.0)
            self.content_loss = \
                PerceptualLoss('PerceptualLoss_content',
                               self.my_config, self.output, self.target,
                               content_layers=self.my_config['content_layers']
                                                  .split(','))
            self.avg_content_loss = 1e5 * self.content_loss.avg_content_loss
            # self.content_losses = self.content_loss.content_losses

            # Average loss over batch
            self.loss = (self.avg_content_loss + self.avg_style_loss) / \
                tf.cast(self.batch_size, tf.float32)

            self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0, tf.uint8),
                             max_outputs=1)
            tf.summary.image('Content_Target',
                             tf.cast(self.target * 255.0, tf.uint8),
                             max_outputs=1)
            # tf.summary.image('Style_Target',
            #                  tf.cast(self.style_target * 255.0, tf.uint8),
            #                  max_outputs=1)

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)
            tf.summary.scalar('Style_Loss', self.avg_style_loss)
            tf.summary.scalar('Content_Loss', self.avg_content_loss)

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
                check_snapshots(self.my_config['run_id'])
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
                create_meshgrid_numpy(225, 225, -112, 112, -112, 112)

            i = 0
            for theta in np.linspace(0, 2*np.pi, 1):
                latent_vector = np.array([[1, 0]],
                                         dtype='float32')
                feed_dict = {self.input_coord: input_coord,
                             self.latent_vector_feed: latent_vector}

                output = sess.run(self.output, feed_dict=feed_dict)
                target_path = os.path.join(self.my_config['data_dir'],
                                           'out', 'predict')
                write_images(output, target_path, training_iteration=i)
                i += 1
