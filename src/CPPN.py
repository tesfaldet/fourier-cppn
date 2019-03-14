import tensorflow as tf
import numpy as np
from src.layers.ConvLayer import ConvLayer
from src.utils.create_meshgrid import create_meshgrid
from src.utils.check_snapshots import check_snapshots
from src.InceptionV1 import InceptionV1


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
        with tf.name_scope('Meshgrid'):
            # TODO: clean this up
            dimensions = self.my_config['dimensions'].split(',')
            self.width = int(dimensions[0])
            self.height = int(dimensions[1])
            r = eval(self.my_config['cppn_coordinate_limit'])  # sketchy af
            self.input = create_meshgrid(self.width, self.height, -r, r, -r, r)

        # CPPN OUTPUT
        with tf.name_scope('CPPN'):
            self.cppn_layers = [('input', self.input)]
            for i in range(self.my_config['cppn_num_layers']):
                prev_layer = self.cppn_layers[i][1]
                prev_num_channels = tf.shape(prev_layer)[-1]
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
            self.output = ConvLayer('output', self.cppn_layers[-1][1], 3,
                                    activation='sigmoid',
                                    weight_init=tf.zeros_initializer())

        # OBJECTIVE
        # self.loss = -InceptionV1('InceptionV1Loss', self.output)\
            # .avg_channel("mixed4b_3x3_pre_relu", 77)

        # self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0,
                                               tf.uint8), max_outputs=6)

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)

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
            writer = tf.summary.FileWriter('logs/' + self.my_config['run_id'],
                                           sess.graph)

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
                    saver.save(sess, 'snapshots/' +
                               self.my_config['run_id'] + '/' +
                               'snapshot_iter', global_step=i)

    def validate(self):
        pass

    def predict(self):
        pass
