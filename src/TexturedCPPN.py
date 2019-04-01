import os
import tensorflow as tf
import numpy as np
from src.utils.check_snapshots import check_snapshots
from src.CPPN import CPPN
from src.TextureMappingNetwork import TextureMappingNetwork
from src.PerceptualLoss import PerceptualLoss
from src.utils.load_image import load_image


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
        self.texture_mapping_network = \
            TextureMappingNetwork(my_config=self.my_config)

        # Combine output from cppn and texture mapping network
        self.output = tf.clip_by_value(self.cppn.output +
                                       self.texture_mapping_network.output,
                                       0., 1.)

        # OBJECTIVE
        target_path = os.path.join(self.my_config['data_dir'],
                                   'textures', 'pebbles.jpg')
        # TODO: clean this up
        dimensions = self.my_config['dimensions'].split(',')
        width = int(dimensions[0])
        height = int(dimensions[1])
        self.target = tf.image.resize_images(
            load_image(target_path), [width, height])
        self.loss = 1e5 * \
            PerceptualLoss(self.my_config, self.output,
                           self.target,
                           style_layers=self.my_config['style_layers']
                                            .split(',')).style_loss

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Final output and target
            tf.summary.image('CPPN+Texture', tf.cast(self.output * 255.0,
                                                     tf.uint8))
            tf.summary.image('Target', tf.cast(self.target * 255.0, tf.uint8))

            # CPPN output and texture mapping network output
            tf.summary.image('CPPN', tf.cast(self.cppn.output * 255.0,
                                             tf.uint8))
            tf.summary.image('Texture',
                             tf.cast((
                                 (self.texture_mapping_network.output + 1.) / 2.) * 255.0, tf.uint8))  # to shift texture mapping output from -1, 1 to 0, 1

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
