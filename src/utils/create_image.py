import numpy as np


def plaid_pattern(width, freq):
    X = np.arange(0, 2 * np.pi, (2 * np.pi)/width).astype('float32')
    X_len = X.size

    X = np.tile(X, X_len).reshape((X_len, X_len))
    Y = X.T

    pattern = np.cos(X * freq) + np.cos(Y * freq)
    # transform to range [0, 1]
    pattern = (pattern - np.min(pattern)) / np.max(pattern - np.min(pattern))

    pattern = np.expand_dims(pattern, 2)  # add channel dimension
    pattern = np.tile(pattern, (1, 1, 3))  # make 3 channel

    pattern = np.expand_dims(pattern, 0)  # add batch dimension

    return pattern


def cos_pattern_horizontal(width, freq):
    X = np.arange(0, 2 * np.pi, (2 * np.pi)/width).astype('float32')
    X_len = X.size

    X = np.tile(X, X_len).reshape((X_len, X_len))

    pattern = np.cos(X * freq)
    # transform to range [0, 1]
    pattern = (pattern - np.min(pattern)) / np.max(pattern - np.min(pattern))

    pattern = np.expand_dims(pattern, 2)  # add channel dimension
    pattern = np.tile(pattern, (1, 1, 3))  # make 3 channel

    pattern = np.expand_dims(pattern, 0)  # add batch dimension

    return pattern


def cos_pattern_vertical(width, freq):
    X = np.arange(0, 2 * np.pi, (2 * np.pi)/width).astype('float32')
    X_len = X.size

    Y = (np.tile(X, X_len).reshape((X_len, X_len))).T

    pattern = np.cos(Y * freq)
    # transform to range [0, 1]
    pattern = (pattern - np.min(pattern)) / np.max(pattern - np.min(pattern))

    pattern = np.expand_dims(pattern, 2)  # add channel dimension
    pattern = np.tile(pattern, (1, 1, 3))  # make 3 channel

    pattern = np.expand_dims(pattern, 0)  # add batch dimension

    return pattern