import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm_notebook
import pickle
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

from azureml.core import Run, Experiment, Workspace, Datastore, Dataset

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)


def create_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by batch_size
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print("The average loss for epoch {} is {:7.2f} ".format(epoch, logs["loss"]))
        run.log("loss", logs["loss"])
        run.log("train_accuracy", logs["accuracy"])
        run.log("val_accuracy", logs["val_accuracy"])
        parent_run.log("loss", logs["loss"])
        parent_run.log("train_accuracy", logs["accuracy"])
        parent_run.log("val_accuracy", logs["val_accuracy"])


parser = argparse.ArgumentParser()
parser.add_argument("--sample_data_size", type=int, required=False, default=-1)
parser.add_argument("--input_data", type=str, required=True)
parser.add_argument("--epochs", type=int, required=False, default=20)
parser.add_argument("--val_size", type=float, required=False, default=0.1)
parser.add_argument("--output_model", type=str, required=True)

args = parser.parse_args()
run = Run.get_context()
parent_run = run.parent
params = {
    "batch_size": 128,  # 20<16<10, 25 was a bust
    "epochs": args.epochs,
    "lr": 0.00010000,
    "time_steps": 100,
    "verbose": 2,
    "sample_data_size": args.sample_data_size,
    "val_size": args.val_size,
}


print("files in the input dolfer:{}".format(os.listdir(args.input_data)))


file_path_x = os.path.join(args.input_data, "X_train_data.npy")
file_path_y = os.path.join(args.input_data, "y_train_data.npy")

x_full = np.load(file_path_x)
y_full = np.load(file_path_y)

y_full += 1
print("label size:", y_full.shape)
y_full = to_categorical(y_full)
print("one hot label size:", y_full.shape)

X_train, X_val, y_train, y_val = train_test_split(
    x_full, y_full, test_size=params["val_size"], shuffle=False
)
print(
    "shapes of X_train {}, y_train, {}, X_val {}, y_val {}".format(
        X_train.shape, y_train.shape, X_val.shape, y_val.shape
    )
)


# Some trimming:
X_train = X_train[: params["sample_data_size"]]
y_train = y_train[: params["sample_data_size"]]
X_train = trim_dataset(X_train, params["batch_size"])
y_train = trim_dataset(y_train, params["batch_size"])

# Throwing away the starting steps to avoid leakage
X_val = X_val[params["time_steps"] : params["sample_data_size"]]
y_val = y_val[params["time_steps"] : params["sample_data_size"]]

X_val = trim_dataset(X_val, params["batch_size"])
y_val = trim_dataset(y_val, params["batch_size"])


print(
    "shapes of X_train {}, y_train, {}, X_val {}, y_val {}".format(
        X_train.shape, y_train.shape, X_val.shape, y_val.shape
    )
)


model = create_model(params["time_steps"], X_train.shape[2], y_train.shape[1])
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    verbose=params["verbose"],
    callbacks=[LossAndErrorPrintingCallback()],
)
os.makedirs(args.output_model, exist_ok=True)
model.save(os.path.join(args.output_model, "model"))
