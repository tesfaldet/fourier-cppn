import tensorflow as tf
import numpy as np
from src.layers.ConvLayer import ConvLayer
from src.utils.create_meshgrid import create_meshgrid
from src.utils.check_snapshots import check_snapshots
from src.InceptionV1 import InceptionV1


class CPPN:
    def __init__(self,
                 my_config,
                 tf_config,
                 target):
        self.tf_config = tf_config
        self.my_config = my_config
        self.target = target
        self.build_graph()
        print('Num Variables: ',
              np.sum([np.product([xi.value for xi in x.get_shape()])
                      for x in tf.all_variables()]))

    def build_graph(self):
        # with tf.name_scope('Target'):
        #     self.target = tf.expand_dims(tf.constant(self.target), axis=0)
        #     image_shape = self.target.get_shape().as_list()
        #     self.width = image_shape[1]
        #     self.height = image_shape[2]

        # COORDINATE MESHGRID INPUT
        with tf.name_scope('Input'):
            self.width = 224
            self.height = 224
            r = 3.0**0.5  # std(coord_range) == 1.0
            self.input = create_meshgrid(self.width, self.height, -r, r, -r, r)

        # CPPN OUTPUT
        with tf.name_scope('CPPN'):
            self.fc1 = ConvLayer('fc1', self.input, 24, activation='atan')
            self.fc2 = ConvLayer('fc2', self.fc1, 24, activation='atan')
            self.fc3 = ConvLayer('fc3', self.fc2, 24, activation='atan')
            self.fc4 = ConvLayer('fc4', self.fc3, 24, activation='atan')
            self.fc5 = ConvLayer('fc5', self.fc4, 24, activation='atan')
            self.fc6 = ConvLayer('fc6', self.fc5, 24, activation='atan')
            self.fc7 = ConvLayer('fc7', self.fc6, 24, activation='atan')
            self.fc8 = ConvLayer('fc8', self.fc7, 24, activation='atan')
            self.output = ConvLayer('output', self.fc8, 3, zeros=True,
                                    activation='sigmoid')

        # OBJECTIVE
        # with tf.name_scope('Loss'):
            # self.loss = tf.nn.l2_loss(self.output - self.target) \
                # / tf.to_float(self.my_config['batch_size'])
        self.loss = -InceptionV1('InceptionV1Loss', self.output)\
            .avg_channel("mixed4b_3x3_pre_relu", 77)

        self.build_summaries()

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Output and Target
            tf.summary.image('Output', tf.cast(self.output * 255.0,
                                               tf.uint8), max_outputs=6)
            # tf.summary.image('Target', tf.cast(self.target * 255.0,
                                            #    tf.uint8), max_outputs=6)

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
