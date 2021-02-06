import numpy as np


def getWeights_FFD(d=0.1, thres=1e-5):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def fracDiff_FFD(mat, d, thres=1e-5):
    # this function is not defined, read text to see what exactly he was trying to do
    #     w = getWeights_FFD(d, thres)
    w = getWeights_FFD(d, thres)
    print("shape w:", w.shape)
    width = len(w)
    dim_0 = mat.shape[0] - len(w) + 1
    dim_1 = mat.shape[1]
    # print(dim_0, dim_1)
    ffd = np.zeros((dim_0, dim_1))
    # print("dim_0", dim_0)
    for i in range(dim_0):
        ffd[i] = np.dot(w.T, mat[i : width + i])
    print("shape of ffd mat {}".format(ffd.shape))
    return ffd


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
    # print("dim_0", dim_0)
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