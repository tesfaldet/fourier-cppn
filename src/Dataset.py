import os
import tensorflow as tf
import multiprocessing
from src.utils.create_meshgrid import create_meshgrid


class Dataset:

    def __init__(self, training_path, config):
        self.train_path = training_path
        self.config = config
        self.handle = tf.placeholder(tf.string, shape=[])

        # input coord, latent vector, target, index
        self.iterator = \
            tf.data.Iterator \
              .from_string_handle(self.handle,
                                  (tf.float32, tf.float32,
                                   tf.int32, tf.string))

        train_generator = self.data_generator(self.train_path)
        self.train_dataset = \
            tf.data.Dataset.from_tensor_slices(train_generator)

        self.num_cores = multiprocessing.cpu_count()
        self.buffer_size = 10

    def get_training_handle(self):
        shuffle_and_repeat = tf.data.experimental.shuffle_and_repeat
        map_and_batch = tf.data.experimental.map_and_batch
        self.train_dataset = self.train_dataset \
            .apply(shuffle_and_repeat(buffer_size=self.buffer_size)) \
            .apply(map_and_batch(self.read_data,
                                 self.config['batch_size'],
                                 num_parallel_batches=self.num_cores)) \
            .prefetch(1)
        self.train_iterator = self.train_dataset.make_one_shot_iterator()
        return self.train_iterator.string_handle()

    def data_generator(self, path):
        filenames = [os.path.join(path, f) for f in sorted(os.listdir(path))]
        num_files = len(filenames)
        input_coords = tf.constant([-1] * num_files)
        filenames = tf.constant(filenames)
        indices = list(range(num_files))
        indices = tf.constant(indices)
        return (input_coords, filenames, indices)

    def read_data(self, input_coord, filename, index):
        # READ TARGET IMAGE
        image_string = tf.read_file(filename)
        image_decoded = \
            tf.cast(tf.image.decode_image(image_string, channels=3),
                    tf.float32)
        image_decoded /= 255.0  # [0, 255] -> [0, 1]
        image_decoded.set_shape([None, None, 3])
        input_dimensions = \
            self.config['cppn_input_dimensions'].split(',')
        input_width = int(input_dimensions[0])
        input_height = int(input_dimensions[1])
        # H x W x 3
        image_decoded = tf.image.resize_images(image_decoded,
                                               size=[input_height,
                                                     input_width])

        # GENERATE INPUT COORDS BASED ON TARGET IMAGE
        input_coord_range = \
            self.config['cppn_input_coordinate_range'].split(',')
        input_coord_min = eval(input_coord_range[0])
        input_coord_max = eval(input_coord_range[1])
        # H x W x 2
        input_coord = \
            create_meshgrid(input_width, input_height,
                            input_coord_min, input_coord_max,
                            input_coord_min, input_coord_max,
                            batch_size=tf.constant(0))

        return input_coord, image_decoded, index, filename
