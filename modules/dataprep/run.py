import multiprocessing as mp
import argparse
import os
import time
import numpy as np
import pandas as pd
import logging
from utils import (
    merge_simultanous_rows,
    save_to_blob,
    standarize,
    normalize,
    scale,
    select_first_file,
    set_df_labels
)
from azureml.core import Run, Experiment, Workspace, Datastore, Dataset
from datetime import datetime, timedelta
from time import sleep
from tqdm import tqdm

def init():
    global default_datastore
    # global trades_dataset_mountpoint
    # df_har_data = pd.read_csv(path_to_har)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_level", default="0.0001")
    parser.add_argument("--sl_level", default="0.0001")
    parser.add_argument("--vol_tick", type=int, default="1000")
    parser.add_argument("--wait_time", type=int, default="600",
                        help="time delta in seconds")
    parser.add_argument(
        "--quotes_path",
        type=str,
        required=False,
        help="raw file",
        default = "datasets/quotes_merged_all"
    )
    parser.add_argument(
        "--trades_path",
        type=str,
        required=False,
        default='datasets/trades_merged_test',
        help="raw file",
    )
    parser.add_argument(
        "--local_data",
        type=bool,
        required=False,
        default=False,
        help="raw file",
    )
    parser.add_argument(
        "--data_output",
        type=str,
        required=False,
        default='datasets/trades_merged_all',
        help="path to merged trades files in Data Store",
    )
    parser.add_argument("--sample_days", type=int, required=False, default=-1)
    global args
    args, _ = parser.parse_known_args()

    run = Run.get_context()
    if "_OfflineRun" in str(run):
        ws = Workspace.from_config()
    else:
        ws = Run.get_context().experiment.workspace
    if not args.local_data:
        default_datastore = ws.get_default_datastore()
        trades_dataset_mount = Dataset.File.from_files((default_datastore, args.trades_path)).mount()
        trades_dataset_mount.start()
        trades_dataset_mountpoint = trades_dataset_mount.mount_point
    else:
        trades_dataset_mountpoint = args.trades_path

def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    start = time.time()
    resultList = []

    for file_name in mini_batch:
        # read each file
        print("******  Processing {}".format(os.path.basename(file_name)))
        df_merged = pd.read_csv(file_name)
        print(df_merged.shape)
        # fixing the volatility, the value should have been squere rooted 
        # but the squared valued got scaled by mistake
        mean_old = 10848.408438877028 # From HAR df
        mean_sqrt = 94.85532420082409 # From HAR df
        df_merged['dailyVolatility'] = df_merged['dailyVolatility'].apply(lambda x:np.sqrt(x*mean_old)/mean_sqrt)

        df_merged['Date-Time'] = pd.to_datetime(df_merged['Date-Time'])

        # reading the corresponding trades file
        # date_string = str(merged_df.loc[0, 'Date-Time'])[:10]
        trades_file_name = os.path.basename(file_name).replace('mq', 'mt')
        try:
            df_merged_trades = pd.read_csv(os.path.join(trades_dataset_mountpoint, trades_file_name))
            print("loaded the corresponding trades file from {}".format(
                os.path.join(trades_dataset_mountpoint, trades_file_name)))
            df_merged_trades['Date-Time'] = pd.to_datetime(df_merged_trades['Date-Time'])
        except Exception as e:
            print('corresponding trades file for {} not found'.format(trades_file_name))
            resultList.append("{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
                os.path.basename(file_name), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            return resultList
        # calulating the tick indices
        vol = df_merged_trades['Acc Volume']
        vol_tick = args.vol_tick
        vol_levels = range(vol_tick, int(max(vol)), vol_tick)
        tick_times = []
        for level in vol_levels:
            ind = vol[vol > level].index.min()
            tick_times.append(df_merged_trades.loc[ind, 'Date-Time'])

        times = df_merged['Date-Time']
        inds = []
        for tick_time in tick_times:
            ind = times[times > tick_time].index.min()
            inds.append(ind)
        inds = [x for x in inds if x > 0]  # remove nans
        print("The indices for ticks:\n {}".format(inds))

        df_merged.dropna(inplace=True)
        print("len inds:{}".format(len(inds)))
        df_bars = set_df_labels(
            df_merged,
            pt_level=float(args.pt_level),
            sl_level=float(args.sl_level),
            wait_time=timedelta(seconds=int(args.wait_time)),
            inds=inds,
        )
        # print('final col names {}'.format(df_bars.columns))
        # df_1day_merged.to_csv(os.path.join('../data/test/', os.path.basename(raw_file_name))[:-3])
        num_pt_long = len(df_bars[df_bars['long_label'] == 1])
        num_sl_long = len(df_bars[df_bars['long_label'] == -1])
        num_te_long = len(df_bars[df_bars['long_label'] == 0])
        num_pt_short = len(df_bars[df_bars['short_label'] == 1])
        num_sl_short = len(df_bars[df_bars['short_label'] == -1])
        num_te_short = len(df_bars[df_bars['short_label'] == 0])
        print("{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
            os.path.basename(file_name), df_bars.shape, num_pt_long, num_sl_long, num_te_long, num_pt_short, num_sl_short, num_te_short))
        # save_to_blob(
        #     df=df_bars,
        #     datastore=default_datastore,
        #     path="datasets/df_bars_test",
        #     file_name="bars_{}".format(os.path.basename(file_name)),
        # )
        print("creating folder {}".format(os.path.dirname(args.data_output)))
        os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
        file_name="bars_{}".format(os.path.basename(file_name))
        df_bars.to_csv(os.path.join(args.data_output, file_name), index=False)
        
        result="{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
            os.path.basename(file_name), df_bars.shape, num_pt_long, num_sl_long, num_te_long, num_pt_short, num_sl_short, num_te_short)
        end = time.time()
        print("{0} was labeled in :{1:5.1f}".format(file_name, end - start))
        with open("labels_logs_{}_{}.log".format(args.pt_level, args.vol_tick),'a') as f:
            f.write(result+'\n')
    
    return result


if __name__ == "__main__":

    init()
    if not args.local_data:
        path_on_datastore = args.quotes_path
        trades_raw_dataset = Dataset.File.from_files(
            (
                default_datastore,
                path_on_datastore,
            ),
            validate=True,
            ).mount()
        trades_raw_dataset.start()
        files_folder = trades_raw_dataset.mount_point
        files_list = os.listdir(files_folder)
    else:
        files_folder = args.quotes_path
        files_list = os.listdir(files_folder)
    print("there are {} files in the input dataset".format(len(files_list)))
    # files_list = [name for name in files_list if (name<'2018-01-01' or name>='2019-01-01')]
    # print("there are {} 2018 files in the input dataset".format(len(files_list)))

    # logs_file = open("labels_logs_sample.log",'w')

    files_folder = args.quotes_path
    input_files = sorted(os.listdir(files_folder))
    input_files = [os.path.join(files_folder, f) for f in input_files]
    print("number of data files found in the input folder: {}".format(len(input_files)))
    input_files = input_files[: args.sample_days]
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(run, [[name] for name in files_list])
    # for i in tqdm(range(len(files_list))):
    #     results = run([os.path.join(files_folder,files_list[i])])
        # logs_file.write(results+'\n')
    # logs_file.close()
    if not args.local_data:
        trades_raw_dataset.stop()



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
        if df_1day_bars.shape[0] == 0:
            resultList.append("{}, {}".format(0, os.path.basename(raw_file_name)))
            return resultList
        #         print(df_1day_bars.dtypes)
        date_string = df_1day_bars.loc[0, "Date-Time"][:10]
        # remove unwanted colls
        base_train_cols = ["Bid Price", "Ask Price", "Bid Size", "Ask Size"]
        label_col = ["long_label"]

        # Adding new columns
        df_1day_bars["BidAskRatio"] = (
            df_1day_bars["Bid Size"] / df_1day_bars["Ask Size"]
        )
        new_train_cols = [f"{col}_diff" for col in base_train_cols]
        df_1day_bars[new_train_cols] = df_1day_bars[base_train_cols].diff()
        train_cols = base_train_cols + new_train_cols + ["BidAskRatio"]
        # df_1day_bars[[f'{col}_fd' for col in train_cols]] = fracDiff_FFD(df_1day_bars[train_cols], 0.35)
        df_1day_bars = df_1day_bars[train_cols + label_col]
        # remove rows with NA values
        df_1day_bars.dropna(inplace=True, axis=0)
        print(
            "shape of data after removing unwanted rows/cols: {}".format(
                df_1day_bars.shape
            )
        )
        print("included columns in the training data: {}".format(train_cols))

        x = df_1day_bars.loc[:, train_cols].values
        y = df_1day_bars.loc[:, label_col].values
        min_max_scaler = MinMaxScaler()
        x_s = min_max_scaler.fit_transform(x)
        print(
            "Deleting unused dataframes of total size(KB)",
            (sys.getsizeof(df_1day_bars)) // 1024,
        )
        del df_1day_bars
        print(
            "Are any NaNs present in train/test matrices?",
            np.isnan(x_s).any(),
            np.isnan(x_s).any(),
        )

        x_t, y_t = build_timeseries(x_s, y, TIME_STEPS)
        print("Batch trimmed size", x_t.shape, y_t.shape)
        if len(X_full) == 0:
            X_full = np.array([]).reshape(0, x_t.shape[1], x_t.shape[2])
            print("initiating X_full data. Shape: {}".format(X_full.shape))

        X_full = np.append(X_full, x_t, axis=0)
        y_full = np.append(y_full, y_t, axis=0)
        print("full sizes:", X_full.shape, y_full.shape)

    return X_full, y_full


parser = argparse.ArgumentParser()
parser.add_argument("--sample_days", type=int, required=False, default=-1)
parser.add_argument("--time_steps", type=int, required=False, default=100)
parser.add_argument("--input_data", type=str, required=True)
parser.add_argument("--output_train_data", type=str, required=True)
parser.add_argument("--output_test_data", type=str, required=True)
parser.add_argument("--test_size", type=float, required=False, default=0.2)

args = parser.parse_args()

TIME_STEPS = args.time_steps

bars_folder = args.input_data

input_files = sorted(os.listdir(bars_folder))
input_files = [os.path.join(bars_folder, f) for f in input_files]
print("number of data files found in the input folder: {}".format(len(input_files)))
input_files = input_files[: args.sample_days]

X_full, y_full = run(input_files)

train_file_path_x = os.path.join(args.output_train_data, f"X_train_data.npy")
train_file_path_y = os.path.join(args.output_train_data, f"y_train_data.npy")

test_file_path_x = os.path.join(args.output_test_data, f"X_test_data.npy")
test_file_path_y = os.path.join(args.output_test_data, f"y_test_data.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=args.test_size, shuffle=False
)

os.makedirs(args.output_train_data, exist_ok=True)
with open(train_file_path_x, "wb") as f:
    np.save(f, X_train)
with open(train_file_path_y, "wb") as f:
    np.save(f, y_train)

os.makedirs(args.output_test_data, exist_ok=True)
with open(test_file_path_x, "wb") as f:
    np.save(f, X_test)
with open(test_file_path_y, "wb") as f:
    np.save(f, y_test)
