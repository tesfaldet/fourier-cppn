import os
import tensorflow as tf
from src.FourierCPPN import FourierCPPN
from src.Dataset import Dataset


# USER SETTINGS
my_config = {}
my_config['batch_size'] = 1
my_config['learning_rate'] = 5e-4
my_config['write_frequency'] = 100
my_config['snapshot_frequency'] = 1000
my_config['print_frequency'] = 1
my_config['log_frequency'] = 1
my_config['iterations'] = 100000
my_config['log_dir'] = 'logs'
my_config['snap_dir'] = 'snapshots'
my_config['data_dir'] = 'data'
my_config['dataset_dir'] = 'dataset'
my_config['train'] = True
my_config['force_train_from_scratch'] = True
my_config['use_bfgs'] = True
my_config['run_id'] = 'test'

my_config['cppn_input_dimensions'] = '225,225'
my_config['cppn_input_coordinate_range'] = '-112,112'
my_config['cppn_coord_scale_factor'] = '(3**0.5)/112'
my_config['cppn_num_layers'] = 8
my_config['cppn_num_neurons'] = 24
my_config['cppn_activation'] = 'atan_concat'
my_config['cppn_latent_size'] = 10
my_config['style_layers'] = 'conv1_1/Relu,pool1,pool2,pool3,pool4'
my_config['content_layers'] = 'conv1_1/Relu,pool1,pool2,pool3,pool4'
my_config['cppn_fourier_dimensions'] = '10,10'
my_config['cppn_fourier_coordinate_range'] = '-5,4'

# GPU SETTINGS
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

# BUILD GRAPH
dataset_path = os.path.join(my_config['data_dir'],
                            my_config['dataset_dir'])
dataset = Dataset(training_path=dataset_path, config=my_config)
cppn = FourierCPPN(dataset=dataset,
                   tf_config=tf_config,
                   my_config=my_config)

if not my_config['train']:
    # PREDICT
    cppn.predict(os.path.join(my_config['snap_dir'], '742685'))
else:
    # TRAIN
    cppn.train()
