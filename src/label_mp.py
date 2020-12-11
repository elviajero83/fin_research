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
    global trades_dataset_mountpoint
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
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(run, [[os.path.join(files_folder,name)] for name in files_list])
    # for i in tqdm(range(len(files_list))):
    #     results = run([os.path.join(files_folder,files_list[i])])
        # logs_file.write(results+'\n')
    # logs_file.close()
    if not args.local_data:
        trades_raw_dataset.stop()
