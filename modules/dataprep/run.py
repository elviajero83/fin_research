import multiprocessing as mp
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from utils import select_first_file, set_df_labels, TW_avg
from azureml.core import Run, Experiment, Workspace, Datastore, Dataset
from datetime import datetime, timedelta
from time import sleep
from tqdm import tqdm
from functools import partial


def init():
    # global trades_dataset_mountpoint
    # df_har_data = pd.read_csv(path_to_har)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_level", default="0.0001")
    parser.add_argument("--sl_level", default="0.0001")
    parser.add_argument("--vol_tick", type=int, default="1000")
    parser.add_argument(
        "--wait_time", type=int, default="600", help="time delta in seconds"
    )
    parser.add_argument(
        "--quotes_path",
        type=str,
        required=False,
        help="raw file",
        default="datasets/quotes_merged_all",
    )
    parser.add_argument(
        "--trades_path",
        type=str,
        required=False,
        default="datasets/trades_merged_test",
        help="raw file",
    )
    parser.add_argument(
        "--data_output",
        type=str,
        required=False,
        default="datasets/trades_merged_all",
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
    print("trades_path {}".format(args.trades_path))
    return args


def run(mini_batch, args):
    print(args.trades_path)
    print(f"run method start: {__file__}, run({mini_batch})")
    start = time.time()
    resultList = []

    for file_name in mini_batch:
        # read each file
        print("******  Processing {}".format(os.path.basename(file_name)))
        df_merged = pd.read_csv(file_name)
        date_string = str(df_merged.loc[0, "Date-Time"])[:10]

        # fixing the volatility, the value should have been squere rooted
        # but the squared valued got scaled by mistake
        mean_old = 10848.408438877028  # From HAR df
        mean_sqrt = 94.85532420082409  # From HAR df
        df_merged["dailyVolatility"] = df_merged["dailyVolatility"].apply(
            lambda x: np.sqrt(x * mean_old) / mean_sqrt
        )

        df_merged["Date-Time"] = pd.to_datetime(df_merged["Date-Time"])

        print("full quotes df shape", df_merged.shape)
        df_merged = df_merged[
            df_merged["Date-Time"]
            >= datetime.strptime(f"{date_string} 8: 30 -0600", "%Y-%m-%d %H: %M %z")
        ]
        print(">8:30  quotes df shape", df_merged.shape)
        df_merged = df_merged[
            df_merged["Date-Time"]
            <= datetime.strptime(f"{date_string} 13: 15 -0600", "%Y-%m-%d %H: %M %z")
        ]
        print(">8:30 < 13:15 quotes df shape", df_merged.shape)
        df_merged.reset_index(inplace=True, drop=True)

        # reading the corresponding trades file
        # date_string = str(merged_df.loc[0, 'Date-Time'])[:10]
        trades_file_name = os.path.basename(file_name).replace("mq", "mt")
        trades_file_path = os.path.join(args.trades_path, trades_file_name)
        try:
            df_merged_trades = pd.read_csv(trades_file_path)
            print(
                "loaded the corresponding trades file from {}".format(trades_file_path)
            )
            df_merged_trades["Date-Time"] = pd.to_datetime(
                df_merged_trades["Date-Time"]
            )
            print("full trades df shape", df_merged_trades.shape)
            df_merged_trades = df_merged_trades[
                df_merged_trades["Date-Time"]
                >= datetime.strptime(f"{date_string} 8: 30 -0600", "%Y-%m-%d %H: %M %z")
            ]
            print(">8:30 trades df shape", df_merged_trades.shape)
            df_merged_trades = df_merged_trades[
                df_merged_trades["Date-Time"]
                <= datetime.strptime(
                    f"{date_string} 13: 15 -0600", "%Y-%m-%d %H: %M %z"
                )
            ]
            print(">8:30 < 13:15 trades df shape", df_merged_trades.shape)
            df_merged_trades.reset_index(inplace=True, drop=True)
        except Exception as e:
            print("corresponding trades file for {} not found".format(trades_file_path))
            resultList.append(
                "{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
                    os.path.basename(file_name),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            return resultList
        # calulating the tick indices
        vol = df_merged_trades["Acc Volume"]
        vol_tick = args.vol_tick
        vol_levels = range(int(min(vol)) + vol_tick, int(max(vol)), vol_tick)
        tick_times = []
        for level in vol_levels:
            ind = vol[vol > level].index.min()
            tick_times.append(df_merged_trades.loc[ind, "Date-Time"])
        print("The tick_times for ticks(5):\n {}".format(tick_times[:5]))
        times = df_merged["Date-Time"]
        inds = []
        for tick_time in tick_times:
            ind = times[times > tick_time].index.min()
            inds.append(ind)
        inds = [x for x in inds if x > 0]  # remove nans
        print("The indices for ticks (5):\n {}".format(inds[:5]))
        inds = sorted(list(set(inds)))
        inds_enclosed = [0] + inds
        # inds_timestamps = df_merged.loc[inds, "Date-Time"].values
        # print("The timestmps for indices for ticks:\n {}".format(inds_timestamps[:5]))
        df_merged.dropna(inplace=True)
        print("len inds:{}".format(len(inds)))
        keys = ["Ask Price", "Ask Size", "Bid Price", "Bid Size"]
        df_TW_avg = TW_avg(
            input_df=df_merged,
            datetime_col="Date-Time",
            keys=keys,
            timestamp_cutoffs=df_merged.loc[
                df_merged.index[inds_enclosed], "Date-Time"
            ].values,
            fillforward=True,
        )
        print("df_TW_avg.shape", df_TW_avg.shape)
        print(df_TW_avg.head())

        df_bars = set_df_labels(
            df_merged,
            pt_level=float(args.pt_level),
            sl_level=float(args.sl_level),
            wait_time=timedelta(seconds=int(args.wait_time)),
            inds=inds,
        )
        print(
            "Deleting unused quotes dataframes of total size {} (KB)".format(
                (sys.getsizeof(df_merged_trades)) // 1024
            )
        )
        del df_merged_trades
        print(
            "Deleting unused trades dataframes of total size {} (KB)".format(
                (sys.getsizeof(df_merged)) // 1024,
            )
        )
        del df_merged

        print("df_bars.shape", df_bars.shape)
        TW_avg_keys = ["TW Avg " + key for key in keys]
        df_bars = pd.concat([df_bars, df_TW_avg[TW_avg_keys]], axis=1)

        # print('final col names {}'.format(df_bars.columns))
        # df_1day_merged.to_csv(os.path.join('../data/test/', os.path.basename(raw_file_name))[:-3])
        num_pt_long = len(df_bars[df_bars["long_label"] == 1])
        num_sl_long = len(df_bars[df_bars["long_label"] == -1])
        num_te_long = len(df_bars[df_bars["long_label"] == 0])
        num_pt_short = len(df_bars[df_bars["short_label"] == 1])
        num_sl_short = len(df_bars[df_bars["short_label"] == -1])
        num_te_short = len(df_bars[df_bars["short_label"] == 0])
        print(
            "{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
                os.path.basename(file_name),
                df_bars.shape,
                num_pt_long,
                num_sl_long,
                num_te_long,
                num_pt_short,
                num_sl_short,
                num_te_short,
            )
        )
        print("creating folder {}".format(os.path.dirname(args.data_output)))
        os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
        file_name = "bars_{}".format(os.path.basename(file_name))
        df_bars.to_csv(os.path.join(args.data_output, file_name), index=False)

        result = "{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
            os.path.basename(file_name),
            df_bars.shape,
            num_pt_long,
            num_sl_long,
            num_te_long,
            num_pt_short,
            num_sl_short,
            num_te_short,
        )
        end = time.time()
        print("{0} was labeled in :{1:5.1f}".format(file_name, end - start))
        os.makedirs("outputs", exist_ok=True)
        with open(
            "outputs/labels_logs_{}_{}.log".format(args.pt_level, args.vol_tick), "a"
        ) as f:
            f.write(result + "\n")

    return result


if __name__ == "__main__":
    args = init()
    files_folder = args.quotes_path
    files_list = os.listdir(files_folder)
    print("there are {} files in the input dataset".format(len(files_list)))

    files_folder = args.quotes_path
    input_files = sorted(os.listdir(files_folder))
    input_files = [os.path.join(files_folder, f) for f in input_files]
    print("number of data files found in the input folder: {}".format(len(input_files)))
    if args.sample_days != -1:
        input_files = input_files[: args.sample_days]
    print("input_files", args.sample_days, input_files)
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(partial(run, args=args), [[name] for name in input_files])
