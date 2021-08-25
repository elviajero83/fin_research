import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import logging
from azureml.core import Run, Experiment, Workspace, Datastore, Dataset
from utils import build_timeseries, trim_dataset, fracDiff_FFD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import durationstr2seconds


def run(mini_batch):
    #     print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    X_full = np.array([])
    y_full = np.array([])

    # for raw_file_name in mini_batch:
    for raw_file_name in tqdm(mini_batch):
        # read each file
        print("******  Processing {}".format(os.path.basename(raw_file_name)))
        df_1day_bars = pd.read_csv(raw_file_name)
        print("original shape of data: {}".format(df_1day_bars.shape))
        # print(df_1day_bars.head())
        if df_1day_bars.shape[0] == 0:
            resultList.append("{}, {}".format(0, os.path.basename(raw_file_name)))
            return resultList
        #         print(df_1day_bars.dtypes)
        # remove unwanted colls
        train_cols = [
            "Ask Price",
            "Ask Size",
            "Bid Price",
            "Bid Size",
            "Spread",
            "Mid Quote",
            "Smart Price",
            "Quote Imbalance",
            "Ask Price_diff_1",
            "Ask Size_diff_1",
            "Bid Price_diff_1",
            "Bid Size_diff_1",
            "Spread_diff_1",
            "Mid Quote_diff_1",
            "Smart Price_diff_1",
            "Quote Imbalance_diff_1",
            "TW Avg Ask Price",
            "TW Avg Ask Size",
            "TW Avg Bid Price",
            "TW Avg Bid Size",
            "TW Avg Spread",
            "TW Avg Mid Quote",
            "TW Avg Smart Price",
            "TW Avg Quote Imbalance",
            "dailyVolatility",
            "BidAskRatio",
            "TW Avg BidAskRatio",
            "Order Imbalance",
            "Close Price",
            "Avg Trade Size",
            "Net SellBuy Count",
        ]

        # label_cols
        base_label_cols = [
            "long_label",
            "long_return",
            "long_duration",
            "pt_long_ind",
            "sl_long_ind",
            "short_label",
            "short_return",
            "short_duration",
            "pt_short_ind",
            "sl_short_ind",
            "end_ind",
        ]
        label_cols_list = [[f"{key}_{i+1}" for key in base_label_cols] for i in range(3)]
        label_cols = label_cols_list[0] + label_cols_list[1] + label_cols_list[2]

        df_1day_bars = df_1day_bars[train_cols + label_cols + ["tick_duration"]]
        duration_cols = (
            [f"long_duration_{i+1}" for i in range(3)]
            + [f"short_duration_{i+1}" for i in range(3)]
            + ["tick_duration"]
        )
        for col in duration_cols:
            # print("converting {}".format(col))
            df_1day_bars[col] = df_1day_bars[col].apply(durationstr2seconds)
        # print(df_1day_bars.head())
        # # remove rows with NA values
        df_1day_bars.dropna(inplace=True, axis=0, subset=train_cols)
        print("shape of data after removing unwanted rows/cols: {}".format(df_1day_bars.shape))
        print("included columns in the training data: {}".format(train_cols))

        x = df_1day_bars.loc[:, train_cols].values
        y = df_1day_bars.loc[:, label_cols].values
        w = df_1day_bars.loc[:, ["tick_duration"]].values
        print("x and y shapes", x.shape, y.shape)
        print(
            "Deleting unused dataframes of total size(KB)",
            (sys.getsizeof(df_1day_bars)) // 1024,
        )
        del df_1day_bars
        print(
            "Are any NaNs present in train/test matrices?",
            np.isnan(x).sum(),
            np.isnan(y).sum(),
            np.isnan(w).sum(),
        )

        if len(X_full) == 0:
            X_full = np.array([]).reshape(0, x.shape[1])
            y_full = np.array([]).reshape(0, y.shape[1])
            w_full = np.array([]).reshape(0, w.shape[1])
            print("initiating X_full data. Shape: {}".format(X_full.shape))

        X_full = np.append(X_full, x, axis=0)
        y_full = np.append(y_full, y, axis=0)
        w_full = np.append(w_full, w, axis=0)
        print("full sizes:", X_full.shape, y_full.shape, w_full.shape)

    return X_full, y_full, w_full


parser = argparse.ArgumentParser()
parser.add_argument("--sample_days", type=int, required=False, default=-1)
parser.add_argument("--time_steps", type=int, required=False, default=100)
parser.add_argument("--input_data", type=str, required=True)
parser.add_argument("--output_train_data", type=str, required=True)
parser.add_argument("--output_test_data", type=str, required=True)
parser.add_argument("--output_LSTM_train_data", type=str, required=True)
parser.add_argument("--output_LSTM_test_data", type=str, required=True)
parser.add_argument("--test_size", type=float, required=False, default=0.2)


args = parser.parse_args()

TIME_STEPS = args.time_steps

bars_folder = args.input_data

input_files = sorted(os.listdir(bars_folder))
input_files = [os.path.join(bars_folder, f) for f in input_files]
print("number of data files found in the input folder: {}".format(len(input_files)))
input_files = input_files[: args.sample_days]

X_full, y_full, w_full = run(input_files)


print("appended all {} days, scaling...".format(len(input_files)))
min_max_scaler = MinMaxScaler()
X_full = min_max_scaler.fit_transform(X_full)


ffd = fracDiff_FFD(X_full, 0.35)
X_full = np.append(X_full[-len(ffd) :], ffd, axis=1)
y_full = y_full[-len(ffd) :]
w_full = w_full[-len(ffd) :]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_full, y_full, w_full, test_size=args.test_size, shuffle=False
)
print(
    "X_train size: {}, X_test size: {}, y_train size: {}, y_test size: {}".format(
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
    )
)


train_file_path_x = os.path.join(args.output_train_data, f"X_train_data.npy")
train_file_path_w = os.path.join(args.output_train_data, f"w_train_data.npy")
train_file_path_y_1 = os.path.join(args.output_train_data, f"y_1_train_data.npy")
train_file_path_y_2 = os.path.join(args.output_train_data, f"y_2_train_data.npy")
train_file_path_y_3 = os.path.join(args.output_train_data, f"y_3_train_data.npy")

test_file_path_x = os.path.join(args.output_test_data, f"X_test_data.npy")
test_file_path_w = os.path.join(args.output_test_data, f"w_test_data.npy")
test_file_path_y_1 = os.path.join(args.output_test_data, f"y_1_test_data.npy")
test_file_path_y_2 = os.path.join(args.output_test_data, f"y_2_test_data.npy")
test_file_path_y_3 = os.path.join(args.output_test_data, f"y_3_test_data.npy")

os.makedirs(args.output_train_data, exist_ok=True)
with open(train_file_path_x, "wb") as f:
    np.save(f, X_train)
with open(train_file_path_w, "wb") as f:
    np.save(f, w_train)
with open(train_file_path_y_1, "wb") as f:
    np.save(f, y_train[:, :11])
with open(train_file_path_y_2, "wb") as f:
    np.save(f, y_train[:, 11:22])
with open(train_file_path_y_3, "wb") as f:
    np.save(f, y_train[:, 22:33])

os.makedirs(args.output_test_data, exist_ok=True)
with open(test_file_path_x, "wb") as f:
    np.save(f, X_test)
with open(test_file_path_w, "wb") as f:
    np.save(f, w_test)
with open(test_file_path_y_1, "wb") as f:
    np.save(f, y_test[:, :11])
with open(test_file_path_y_2, "wb") as f:
    np.save(f, y_test[:, 11:22])
with open(test_file_path_y_3, "wb") as f:
    np.save(f, y_test[:, 22:33])

# X_LSTM_train, y_LSTM_train = build_timeseries(X_train, y_train, TIME_STEPS)
# X_LSTM_test, y_LSTM_test = build_timeseries(X_test, y_test, TIME_STEPS)
# print("LSTM train and test sizes", X_LSTM_train.shape, X_LSTM_test.shape)

# train_LSTM_file_path_x = os.path.join(args.output_LSTM_train_data, f"X_train_data.npy")
# train_LSTM_file_path_y = os.path.join(args.output_LSTM_train_data, f"y_train_data.npy")

# test_LSTM_file_path_x = os.path.join(args.output_LSTM_test_data, f"X_test_data.npy")
# test_LSTM_file_path_y = os.path.join(args.output_LSTM_test_data, f"y_test_data.npy")


# os.makedirs(args.output_LSTM_train_data, exist_ok=True)
# with open(train_LSTM_file_path_x, "wb") as f:
#     np.save(f, X_LSTM_train)
# with open(train_LSTM_file_path_y_1, "wb") as f:
#     np.save(f, y_LSTM_train[:, :11])
# with open(train_LSTM_file_path_y_2, "wb") as f:
#     np.save(f, y_LSTM_train[:, 11:22])
# with open(train_LSTM_file_path_y_3, "wb") as f:
#     np.save(f, y_LSTM_train[:, 22:33])

# os.makedirs(args.output_LSTM_test_data, exist_ok=True)
# with open(test_LSTM_file_path_x, "wb") as f:
#     np.save(f, X_LSTM_test)
# with open(test_LSTM_file_path_y_1, "wb") as f:
#     np.save(f, y_LSTM_test[:, :11])
# with open(test_LSTM_file_path_y_2, "wb") as f:
#     np.save(f, y_LSTM_test[:, 11:22])
# with open(test_LSTM_file_path_y_3, "wb") as f:
#     np.save(f, y_LSTM_test[:, 22:33])
