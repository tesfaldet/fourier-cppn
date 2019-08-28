import os
import time
import argparse
import tensorflow as tf
from src.FourierCPPN import FourierCPPN
from src.CPPN import CPPN
from src.Dataset import Dataset


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=100000, type=int)
parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float)
parser.add_argument('-logf', '--log_frequency', default=10, type=int)
parser.add_argument('-printf', '--print_frequency', default=10, type=int)
parser.add_argument('-snapf', '--snapshot_frequency', default=2000, type=int)
parser.add_argument('-writef', '--write_frequency', default=100, type=int)
parser.add_argument('-log_dir', '--log_dir', default='logs', type=str)
parser.add_argument('-snap_dir', '--snapshot_dir',
                    default='snapshots', type=str)
parser.add_argument('-data_dir', '--data_dir', default='data', type=str)
parser.add_argument('-dataset_dir', '--dataset_dir', default='dataset',
                    type=str)
parser.add_argument('-id', '--run_id', default=time.strftime('%d%b-%X'),
                    type=str)
parser.add_argument('-train', '--train', default=True, type=bool)
parser.add_argument('-cppn', '--use_cppn', default=False, type=bool)
parser.add_argument('-bfgs', '--use_bfgs', default=True, type=bool)

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
my_config['dataset_dir'] = args.dataset_dir
my_config['train'] = args.train
my_config['force_train_from_scratch'] = args.force_train_from_scratch
my_config['use_bfgs'] = args.use_bfgs

# GPU SETTINGS
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

# DATASET SETUP
dataset_path = os.path.join(my_config['data_dir'],
                            my_config['dataset_dir'])
dataset = Dataset(training_path=dataset_path, config=my_config)

# BUILD GRAPH
if args.use_cppn:
    cppn = CPPN(dataset=dataset,
                tf_config=tf_config,
                my_config=my_config)
else:
    cppn = FourierCPPN(dataset=dataset,
                       tf_config=tf_config,
                       my_config=my_config)

if not args.train:
    # PREDICT
    cppn.predict(os.path.join(my_config['snap_dir'], '846577'))
else:
    # NOTE KEEPING
    notes_path = os.path.join(my_config['log_dir'],
                              my_config['run_id'] + '.txt')
    with open(notes_path, 'w') as fp:
        notes = \
            '\n'.join('{!s}={!r}'.format(key, val) for (key, val)
                      in my_config.items())
        fp.write(notes)
        fp.write('\n')

    # TRAIN
    cppn.train()
