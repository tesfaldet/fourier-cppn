import tensorflow as tf


def EmbeddingLayer(name, input, embedding_size, dataset_size,
                   weight_init=tf.initializers.truncated_normal(),
                   trainable=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = [dataset_size, embedding_size]
        embeddings = tf.get_variable('image_embeddings',
                                     initializer=weight_init,
                                     shape=shape,
                                     trainable=trainable)
        embedded_image = tf.nn.embedding_lookup(embeddings, input)
        return embedded_image
