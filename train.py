import os
import time
import argparse
import tensorflow as tf
from src.utils.load_image import load_image
from src.CPPN import CPPN


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=5000, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float)
parser.add_argument('-logf', '--log_frequency', default=10, type=int)
parser.add_argument('-printf', '--print_frequency', default=10, type=int)
parser.add_argument('-snapf', '--snapshot_frequency', default=500, type=int)
parser.add_argument('-id', '--run_id', default=time.strftime('%d%b-%X'),
                    type=str)
args = parser.parse_args()

# USER SETTINGS
my_config = {}
my_config['batch_size'] = args.batch_size
my_config['learning_rate'] = args.learning_rate
my_config['snapshot_frequency'] = args.snapshot_frequency
my_config['print_frequency'] = args.print_frequency
my_config['log_frequency'] = args.log_frequency
my_config['run_id'] = args.run_id
my_config['iterations'] = args.iterations

# GPU SETTINGS
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

# DATA INPUT
target = load_image('data/cat.jpg')

# BUILD GRAPH
with tf.device('/gpu:' + str(args.gpu)):
    m = CPPN(tf_config=tf_config, my_config=my_config, target=target)

# TRAIN
m.train()