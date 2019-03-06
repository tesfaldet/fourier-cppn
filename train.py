import os
import time
import argparse
import tensorflow as tf
from src.TexturedCPPN import TexturedCPPN

# SHURIKEN IMPORTS
from shuriken.utils import get_hparams


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=5000, type=int)
parser.add_argument('-lr', '--learning_rate', default=5e-3, type=float)
parser.add_argument('-logf', '--log_frequency', default=10, type=int)
parser.add_argument('-printf', '--print_frequency', default=10, type=int)
parser.add_argument('-snapf', '--snapshot_frequency', default=5000, type=int)
parser.add_argument('-log_dir', '--log_dir', default=".logs", type=str)
parser.add_argument('-snap_dir', '--snapshot_dir', default=".snapshots", type=str)
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
my_config['log_dir'] = args.log_dir
my_config['snap_dir'] = args.snapshot_dir

# SHURIKEN MAGIC
my_config.update(get_hparams())

# GPU SETTINGS
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

# BUILD GRAPH
m = TexturedCPPN(tf_config=tf_config, my_config=my_config)

# TRAIN
m.train()
