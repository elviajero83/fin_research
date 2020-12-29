import numpy as np


def build_timeseries(mat, labels, steps=100):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - steps
    dim_1 = mat.shape[1]
    # print(dim_0, steps, dim_1)
    x = np.zeros((dim_0, steps, dim_1))
    y = np.zeros((dim_0,))
    print("dim_0", dim_0)
    #     for i in tqdm_notebook(range(dim_0)):
    for i in range(dim_0):
        x[i] = mat[i : steps + i]
        y[i] = labels[steps + i]
    #         if i < 10:
    #           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat