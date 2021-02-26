import multiprocessing as mp
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from utils import (
    select_first_file,
    set_df_labels,
    TW_avg,
    data_process,
    trim_df_to_time,
    safe_division,
)
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


def run(file_name, args):
    start = time.time()
    try:
        print("******  Processing {}".format(os.path.basename(file_name)))
        df_quotes = pd.read_csv(file_name)

        # fixing the volatility, the value should have been squere rooted
        # but the squared valued got scaled by mistake
        mean_old = 10848.408438877028  # From HAR df
        mean_sqrt = 94.85532420082409  # From HAR df
        df_quotes["dailyVolatility"] = df_quotes["dailyVolatility"].apply(
            lambda x: np.sqrt(x * mean_old) / mean_sqrt
        )
        print("trimming quotes df")
        df_quotes = trim_df_to_time(df_quotes, "8:30:00.00", "15:00:00.00")

        # reading the corresponding trades file
        # date_string = str(merged_df.loc[0, 'Date-Time'])[:10]
        trades_file_name = os.path.basename(file_name).replace("mq", "mt")
        trades_file_path = os.path.join(args.trades_path, trades_file_name)

        df_trades = pd.read_csv(trades_file_path)
        print("loaded the corresponding trades file from {}".format(trades_file_path))
        df_trades = trim_df_to_time(df_trades, "8:30:00.00", "15:00:00.00")

        # ***************************************************
        # Clean Quotes and Trades
        cleaner = data_process()

        # Quotes
        cleaner.set_negatives_to_nan(
            df_quotes, keys=["Ask Price", "Bid Price", "Ask Size", "Bid Size"]
        )
        cleaner.drop_nans(
            df_quotes, keys=["Ask Price", "Bid Price", "Ask Size", "Bid Size"]
        )

        # Trades
        # Drop zero/negative
        cleaner.set_negatives_to_nan(df_trades, keys=["Price", "Acc Volume"])
        cleaner.drop_nans(df_trades, keys=["Price", "Acc Volume"])

        # Detect outliers
        cleaner.detect_outliers(df_trades, 40, 6, "Price")
        df_trades["Price"] = np.where(df_trades["Outlier"], np.nan, df_trades["Price"])
        df_trades["Acc Volume"] = np.where(
            df_trades["Outlier"], np.nan, df_trades["Acc Volume"]
        )
        cleaner.set_negatives_to_nan(df_trades, keys=["Price", "Acc Volume"])
        cleaner.drop_nans(df_trades, keys=["Price", "Acc Volume"])

        keys = ["Ask Price", "Ask Size", "Bid Price", "Bid Size"]

        # Forward fill quotes
        for key in keys:
            df_quotes[key] = df_quotes[key].fillna(method="ffill")

        # Form lag(price) with distinct value
        df_trades["temp_lag_price"] = df_trades["Price"].shift(1)
        df_trades["Lag(Price) Distinct"] = np.where(
            df_trades["temp_lag_price"] == df_trades["Price"],
            np.NaN,
            df_trades["temp_lag_price"],
        )
        df_trades["Lag(Price) Distinct"].ffill(inplace=True)
        df_trades.drop("temp_lag_price", 1, inplace=True)

        # Merge Quotes and Trades
        df_quotes["Type"] = "Quote"
        df_all = pd.concat(
            [
                df_quotes[["Date-Time", "Type", "Ask Price", "Bid Price", "Seq. No."]],
                df_trades[
                    [
                        "Date-Time",
                        "Price",
                        "Lag(Price) Distinct",
                        "Acc Volume",
                        "Seq. No.",
                    ]
                ],
            ],
            sort=True,
        ).sort_values(by=["Date-Time", "Seq. No."])

        df_all["Type"].fillna(value="Trade", inplace=True)
        df_all["Ask Price"].ffill(inplace=True)
        df_all["Bid Price"].ffill(inplace=True)
        df_trades_all = df_all[df_all["Type"] == "Trade"].copy(deep=True)
        df_trades_all.reset_index(drop=True, inplace=True)

        # Mid Quote
        df_trades_all["Mid Quote"] = (
            df_trades_all["Ask Price"] + df_trades_all["Bid Price"]
        ) / 2

        # Finding Tick Dir (Lee and Ready)
        # Buy
        df_trades_all["case1"] = np.where(
            df_trades_all["Price"] > df_trades_all["Mid Quote"], 1, 0
        )
        # Sell
        df_trades_all["case2"] = np.where(
            df_trades_all["Price"] < df_trades_all["Mid Quote"], -1, 0
        )
        # Buy
        df_trades_all["case3"] = np.where(
            (df_trades_all["Price"] == df_trades_all["Mid Quote"])
            & (df_trades_all["Price"] > df_trades_all["Lag(Price) Distinct"]),
            1,
            0,
        )
        # Sell
        df_trades_all["case4"] = np.where(
            (df_trades_all["Price"] == df_trades_all["Mid Quote"])
            & (df_trades_all["Price"] < df_trades_all["Lag(Price) Distinct"]),
            -1,
            0,
        )

        df_trades_all["Tick Dir"] = (
            df_trades_all["case1"]
            + df_trades_all["case2"]
            + df_trades_all["case3"]
            + df_trades_all["case4"]
        )
        df_trades_all["Signed Trade SQRT"] = df_trades_all["Tick Dir"] * np.sqrt(
            df_trades_all["Acc Volume"]
        )
        df_trades_all["Signed Trade"] = df_trades_all["Tick Dir"] * (
            df_trades_all["Acc Volume"]
        )

        print(df_trades_all.head(30))

        # Calculate some instantanous measures
        df_quotes["Spread"] = df_quotes["Ask Price"] - df_quotes["Bid Price"]
        df_quotes["Mid Quote"] = (df_quotes["Ask Price"] + df_quotes["Bid Price"]) / 2
        df_quotes["Smart Price"] = (
            df_quotes["Ask Price"] * (1 / df_quotes["Ask Size"])
            + df_quotes["Bid Price"] * (1 / df_quotes["Bid Size"])
        ) / (1 / df_quotes["Ask Size"] + 1 / df_quotes["Bid Size"])
        df_quotes["Quote Imbalance"] = np.log(df_quotes["Ask Size"]) - np.log(
            df_quotes["Bid Size"]
        )

        # ***********************************************

        # calulating the tick indices
        vol = df_trades_all["Acc Volume"]
        vol_tick = args.vol_tick
        vol_levels = range(int(min(vol)) + vol_tick, int(max(vol)), vol_tick)
        tick_times = []
        for level in vol_levels:
            ind = vol[vol > level].index.min()
            tick_times.append(df_trades_all.loc[ind, "Date-Time"])
        print("The tick_times for ticks(5):\n {}".format(tick_times[:5]))
        times = df_quotes["Date-Time"]
        inds = []
        for tick_time in tick_times:
            ind = times[times < tick_time].index.max()
            inds.append(ind)
        inds = [x for x in inds if x > 0]  # remove nans
        print("The indices for ticks (5):\n {}".format(inds[:5]))
        inds = sorted(list(set(inds)))
        inds_enclosed = [0] + inds
        # inds_timestamps = df_quotes.loc[inds, "Date-Time"].values
        # print("The timestmps for indices for ticks:\n {}".format(inds_timestamps[:5]))
        df_quotes.dropna(inplace=True)
        print("len inds:{}".format(len(inds)))

        trades_times = df_trades_all["Date-Time"]
        trades_inds = []
        for ind in inds:
            trade_ind = trades_times[
                trades_times > df_quotes.loc[ind, "Date-Time"]
            ].index.min()
            trades_inds.append(trade_ind)
        print("The indices for trades_inds (5):\n {}".format(trades_inds[:5]))
        print("len trades inds:{}".format(len(trades_inds)))
        print(df_trades_all.loc[trades_inds])
        trades_inds_enclosed = [0] + trades_inds
        for i in range(len(trades_inds_enclosed) - 1):
            df_trades_all.loc[
                trades_inds_enclosed[i + 1], "Order Imbalance"
            ] = df_trades_all.loc[
                trades_inds_enclosed[i] + 1 : trades_inds_enclosed[i + 1]
            ][
                "Signed Trade"
            ].sum()
            df_trades_all.loc[
                trades_inds_enclosed[i + 1], "Open Price"
            ] = df_trades_all.loc[trades_inds_enclosed[i], "Price"]
            df_trades_all.loc[
                trades_inds_enclosed[i + 1], "Close Price"
            ] = df_trades_all.loc[trades_inds_enclosed[i + 1], "Price"]
            df_trades_all.loc[
                trades_inds_enclosed[i + 1], "Avg Trade Size"
            ] = df_trades_all.loc[
                trades_inds_enclosed[i] + 1 : trades_inds_enclosed[i + 1]
            ][
                "Acc Volume"
            ].mean()
            df_trades_all.loc[
                trades_inds_enclosed[i + 1], "Net SellBuy Count"
            ] = safe_division(
                sum(
                    df_trades_all.loc[
                        trades_inds_enclosed[i] + 1 : trades_inds_enclosed[i + 1]
                    ]["Tick Dir"]
                    == -1
                ),
                sum(
                    df_trades_all.loc[
                        trades_inds_enclosed[i] + 1 : trades_inds_enclosed[i + 1]
                    ]["Tick Dir"]
                    == 1
                ),
            )
        df_trades_ticks = df_trades_all.loc[trades_inds][
            [
                "Order Imbalance",
                "Close Price",
                "Avg Trade Size",
                "Net SellBuy Count",
            ]
        ]
        df_trades_ticks.reset_index(inplace=True, drop=True)
        print("df_trades_ticks.shape", df_trades_ticks.shape)
        print("df_trades_ticks columns", df_trades_ticks.columns)
        keys = keys + [
            "Spread",
            "Mid Quote",
            "Smart Price",
            "Quote Imbalance",
        ]
        df_TW_avg = TW_avg(
            input_df=df_quotes,
            datetime_col="Date-Time",
            keys=keys,
            timestamp_cutoffs=df_quotes.loc[
                df_quotes.index[inds_enclosed], "Date-Time"
            ].values,
            fillforward=True,
        )
        print("df_TW_avg.shape", df_TW_avg.shape)
        print("df_TW_avg columns", df_TW_avg.columns)
        print(df_TW_avg.head())

        df_bars = set_df_labels(
            df_quotes,
            pt_level=float(args.pt_level),
            sl_level=float(args.sl_level),
            wait_time=timedelta(seconds=int(args.wait_time)),
            inds=inds,
        )
        print(
            "Deleting unused quotes dataframes of total size {}, {},{},{} (KB)".format(
                (sys.getsizeof(df_trades)) // 1024,
                (sys.getsizeof(df_quotes)) // 1024,
                (sys.getsizeof(df_trades_all)) // 1024,
                (sys.getsizeof(df_all)) // 1024,
            )
        )
        del df_trades
        del df_quotes
        del df_trades_all
        del df_all

        print("df_bars.shape", df_bars.shape)
        TW_avg_keys = ["TW Avg " + key for key in keys]
        df_bars = pd.concat([df_bars, df_TW_avg[TW_avg_keys], df_trades_ticks], axis=1)

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

    except Exception as e:
        print(
            "corresponding trades file for {} not found".format(
                os.path.basename(file_name)
            )
        )
        print(e)
        result = "{}, shape: {}, num_pt_long: {}, num_sl_long: {}, num_te_long: {}, num_pt_short: {}, num_sl_short: {}, num_te_short: {}".format(
            os.path.basename(file_name),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    os.makedirs("outputs", exist_ok=True)
    with open(
        "outputs/labels_logs_{}_{}.log".format(args.pt_level, args.vol_tick),
        "a",
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
    results = pool.map(partial(run, args=args), [name for name in input_files])
