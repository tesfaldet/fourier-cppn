import os
import time
import argparse
import tensorflow as tf
import numpy as np
from src.FourierCPPN import FourierCPPN
from src.RGBCPPN import RGBCPPN
from src.utils.write_images import write_images


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=100000, type=int)
parser.add_argument('-lr', '--learning_rate', default=5e-3, type=float)
parser.add_argument('-logf', '--log_frequency', default=10, type=int)
parser.add_argument('-printf', '--print_frequency', default=10, type=int)
parser.add_argument('-snapf', '--snapshot_frequency', default=1000, type=int)
parser.add_argument('-writef', '--write_frequency', default=100, type=int)
parser.add_argument('-log_dir', '--log_dir', default='logs', type=str)
parser.add_argument('-snap_dir', '--snapshot_dir',
                    default='snapshots', type=str)
parser.add_argument('-data_dir', '--data_dir', default='data', type=str)
parser.add_argument('-id', '--run_id', default=time.strftime('%d%b-%X'),
                    type=str)
parser.add_argument('-predict', '--predict', default=True, type=bool)
parser.add_argument('--force_train_from_scratch', default=True, type=bool)
args = parser.parse_args()

# USER SETTINGS
my_config = {}
my_config['batch_size'] = args.batch_size
my_config['learning_rate'] = args.learning_rate
my_config['write_frequency'] = args.write_frequency
my_config['snapshot_frequency'] = args.snapshot_frequency
my_config['print_frequency'] = args.print_frequency
my_config['log_frequency'] = args.log_frequency
my_config['run_id'] = args.run_id
my_config['iterations'] = args.iterations
my_config['log_dir'] = args.log_dir
my_config['snap_dir'] = args.snapshot_dir
my_config['data_dir'] = args.data_dir
my_config['force_train_from_scratch'] = args.force_train_from_scratch
my_config['learning_rate'] = 5e-4
my_config['batch_size'] = 1,
my_config['input_dimensions'] = '224,224'
my_config['target_dimensions'] = '224,224'
my_config['target_image_name'] = 'peppers.jpg'
my_config['log_dir'] = './logs'
my_config['snap_dir'] = './snapshots'
my_config['data_dir'] = './data'
my_config['cppn_input_coordinate_range'] = '-3**0.5,3**0.5'
my_config['cppn_num_layers'] = 8
my_config['cppn_num_neurons'] = 24
my_config['cppn_activation'] = 'atan_concat'
my_config['style_layers'] = 'conv1_1/Relu,pool1,pool2,pool3,pool4'
my_config['cppn_fourier_dimensions'] = '10,10'
my_config['cppn_fourier_coordinate_range'] = '0,9'
my_config['run_id'] = 'local'

# GPU SETTINGS
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

# BUILD GRAPHS
cppn = FourierCPPN(tf_config=tf_config, my_config=my_config)
print('VARIABLES OF GRAPH')
print(tf.trainable_variables())
saver = tf.train.Saver()
with tf.Session(config=tf_config) as sess:
    saver.restore(sess, '/Users/matthewtesfaldet/Projects/texture-cppn/snapshots/616117/snapshot_iter-00016417')

    target_path = os.path.join(my_config['data_dir'], 'out',
                               'predict')

    i = 0
    for theta in np.linspace(0, 2 * np.pi, 200):
        class_1 = np.tile(np.reshape(np.array([0., 1.],
                                              dtype=np.float32),
                                     [1, 1, 1, 2]),
                          [1, cppn.input_height, cppn.input_width, 1])
        class_2 = np.tile(np.reshape(np.array([1., 0.],
                                              dtype=np.float32),
                                     [1, 1, 1, 2]),
                          [1, cppn.input_height, cppn.input_width, 1])
        classes = np.concatenate([class_1, class_2], axis=0)
        feed_dict = {cppn.latent_vector: classes}
        write_images(sess.run(cppn.output, feed_dict=feed_dict), target_path,
                     training_iteration=i)
        i += 1