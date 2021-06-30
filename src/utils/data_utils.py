import numpy as np


def build_timeseries(mat, labels, steps, weights=None):
    """Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.

    Args:
        mat ([type]): ndarray which holds the dataset
        labels ([type]): 1-d which holds the labels
        steps ([type]): [description]
        weights ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: returns two ndarrays
        [type]: returns two ndarrays
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - steps
    dim_1 = mat.shape[1]
    # this size (sample size, steps, n_variables) suited well for time series analysis, e.g. lstm
    x = np.zeros((dim_0, steps, dim_1))
    y = np.zeros((dim_0,))
    w = np.zeros((dim_0,))
    print("dim_0", dim_0)

    for i in range(dim_0):
        x[i] = mat[i : steps + i]
        y[i] = labels[steps + i]
        if weights is not None:
            w[i] = weights[steps + i]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y, w
