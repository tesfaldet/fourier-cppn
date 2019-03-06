import tensorflow as tf


def GramLayer(activations, normalize_method=None):
    # Takes (batches, channels, height, width) and computes gramians of
    # dimension (batches, channels, channels)
    activations_shape = activations.get_shape().as_list()
    """
    Instead of iterating over #channels width by height matrices and computing
    similarity, we vectorize and compute the entire gramian in a single matrix
    multiplication.
    """
    vectorized_activations = tf.reshape(activations,
                                        [activations_shape[0],
                                         activations_shape[1], -1])
    transposed_vectorized_activations = tf.transpose(vectorized_activations,
                                                     perm=[0, 2, 1])
    mult = tf.matmul(vectorized_activations,
                     transposed_vectorized_activations)
    if normalize_method == 'ulyanov':
        # also jcjohnson method
        normalize_scale = tf.div(1.0, (activations_shape[1] *
                                       activations_shape[2] *
                                       activations_shape[3]))
        mult = tf.multiply(normalize_scale, mult)
    elif normalize_method == 'gatys':
        normalize_scale = tf.div(1.0, (2 *
                                       activations_shape[1] *
                                       activations_shape[2] *
                                       activations_shape[3])**2)
        mult = tf.multiply(normalize_scale, mult)

    return mult