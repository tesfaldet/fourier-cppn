import tensorflow as tf
import numpy as np
from src.utils.check_snapshots import check_snapshots
from src.CPPN import CPPN
from src.TextureMappingNetwork import TextureMappingNetwork
from src.InceptionV1 import InceptionV1


class TexturedCPPN:
    def __init__(self,
                 my_config,
                 tf_config):
        self.tf_config = tf_config
        self.my_config = my_config
        self.build_graph()
        print('TexturedCPPN Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.global_variables()]))

    def build_graph(self):
        # Build CPPN
        self.cppn = CPPN(tf_config=self.tf_config, my_config=self.my_config)

        # Build TextureMappingNetwork
        self.texture_mapping_network = TextureMappingNetwork()

        with tf.variable_scope('mixing_params', reuse=tf.AUTO_REUSE):
            self.alpha = tf.get_variable('alpha',
                                         initializer=tf.initializers.constant(0.0),
                                         shape=[])
            self.alpha = tf.nn.sigmoid(self.alpha)

        # Combine output from cppn and texture mapping network
        self.output = 0.5 * (self.alpha * self.cppn.output +
                             (1 - self.alpha) *
                             self.texture_mapping_network.output)

        # OBJECTIVE
        self.loss = -InceptionV1('InceptionV1Loss', self.output)\
            .avg_channel("mixed4b_3x3_pre_relu", 77)

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Combined', tf.cast(self.output * 255.0,
                                                 tf.uint8))
            tf.summary.image('Texture', tf.cast(self.texture_mapping_network.output * 255.0,
                                                tf.uint8))
            tf.summary.image('CPPN', tf.cast(self.cppn.output * 255.0,
                                                tf.uint8))

            tf.summary.scalar('alpha', self.alpha)
            # tf.summary.image('Target', tf.cast(self.target * 255.0,
            #                                      tf.uint8))

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
                    saver.save(sess, self.my_config'snapshots/' +
                               self.my_config['run_id'] + '/' +
                               'snapshot_iter', global_step=i)

    def validate(self):
        pass

    def predict(self):
        pass
