import multiprocessing as mp
import argparse
import os
import numpy as np
import pandas as pd
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
from time import sleep
from tqdm import tqdm


def init():
    parser = argparse.ArgumentParser(
        description="prs for merge"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=False,
        help="raw file",
    )
    parser.add_argument(
        "--data_output",
        type=str,
        required=False,
        default='datasets/trades_merged_all',
        help="path to merged trades files in Data Store",
    )
    parser.add_argument(
        "--sample",
        type=int,
        required=False,
        default=0,
        help="sample df to specific number of lines",
    )
    global args
    args, _ = parser.parse_known_args()

    global default_datastore
    global df_esv_data

    run = Run.get_context()

    if "_OfflineRun" in str(run):
        ws = Workspace.from_config()
    else:
        ws = Run.get_context().experiment.workspace
    default_datastore = ws.get_default_datastore()
    default_datastore = ws.get_default_datastore()
    esv_dataset = Dataset.File.from_files((default_datastore, "datasets/esv"))
    with esv_dataset.mount() as mount_context:
        df_esv_data = pd.read_csv(select_first_file(
            mount_context.mount_point), header=None, names=['Date', 'ESV', 'd1', 'd2', 'd3'])
    print("esv df shape: {}".format(df_esv_data.shape))


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []

    # for raw_file_name in mini_batch:
    for raw_file_name in mini_batch:
        # read each file
        print("******  Processing {}".format(os.path.basename(raw_file_name)))
        df_1day_raw = pd.read_csv(raw_file_name, compression='gzip')
        if df_1day_raw.shape[0] == 0:
            resultList.append("{}, {}, {}".format(os.path.basename(raw_file_name), df_1day_raw.shape, 'None'))
            return resultList
        # pick the right RIC
        print("original shape of data: {}".format(df_1day_raw.shape))
        date_string = df_1day_raw.loc[0, 'Date-Time'][:10]
        today_esv_df = df_esv_data.loc[df_esv_data['Date'] == date_string, 'ESV']
        if len(today_esv_df)>0: 
            today_esv = today_esv_df.values[0].strip()
        else:
            print("No esv data, probably a holiday ")
            resultList.append("{}, {}, {}".format(os.path.basename(raw_file_name), df_1day_raw.shape, 'None'))
            return resultList

        df_1day_raw = df_1day_raw[df_1day_raw['#RIC'] == today_esv]
        df_1day_raw.reset_index(level=0, inplace=True)
        print("shape of data after selecting one RIC: {}".format(df_1day_raw.shape))

        # remove unwanted colls
        drop_cols = ['#RIC', 'Acc. Volume', 'Domain', 'Alias Underlying RIC', 'Type']
        df_1day_raw.drop(columns=drop_cols, inplace=True)

        # sample data
        if args.sample and (df_1day_raw.shape[0] > 0):

            df_1day_raw = df_1day_raw.loc[:int(args.sample), :]
            print("shape of data after sampling: {}".format(
                df_1day_raw.shape))

        # Clean Trade Data- remove rows with negative dollar value
        cleaning_cols = ['Price', 'Volume']
        df_1day_raw[cleaning_cols] = df_1day_raw[cleaning_cols].apply(
            lambda x: np.where(x <= 0, np.nan, x))
        df_1day_raw.dropna(axis=0, inplace=True, subset=cleaning_cols)
        print("shape of data after removing negative  values: {}".format(df_1day_raw.shape))

        df_1day_raw.dropna(axis=0, inplace=True)
        df_1day_raw['Date-Time'] = pd.to_datetime(df_1day_raw['Date-Time'])
        df_1day_merged = df_1day_raw.groupby(
            'Date-Time').apply(merge_simultanous_rows)
        # df_1day_quotes_merged = df_1day_raw.groupby('Date-Time').agg({'Bid Price': 'max',
        #                                                                      'Ask Price': 'min',
        #                                                                      'Bid Size': 'sum',
        #                                                                      'Ask Size': 'sum',
        #                                                                      'Seq. No.': 'min',
        #                                                                      'Exch Time': 'min',
        #                                                                      '#RIC': 'min'})
        print("shape of data after merging simultanous quotes: {}".format(df_1day_merged.shape))

        # calculating ACC Vol
        df_1day_merged["Acc Volume"] = df_1day_merged["Volume"].cumsum()

        # setting index back to numbers
        df_1day_merged.reset_index(level=0, inplace=True)

        # Save the merged data
        # save_to_blob(df=df_1day_merged, datastore=default_datastore, path=args.data_output,
        #              file_name=os.path.basename(raw_file_name).replace('.csv.gz', '-mts.csv' if args.sample else '-mt.csv'))
        print("creating folder {}".format(os.path.dirname(args.data_output)))
        os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
        file_name=os.path.basename(raw_file_name).replace('.csv.gz', '-mts.csv' if args.sample else '-mt.csv')
        df_1day_merged.to_csv(os.path.join(args.data_output, file_name), index=False)

        resultList.append("{}, {}, {}".format(
            os.path.basename(raw_file_name), df_1day_merged.shape, today_esv))
        
    return resultList


if __name__ == "__main__":

    init()
    path_on_datastore = "datasets/trades_raw"
    trades_raw_dataset = Dataset.File.from_files(
        (
            default_datastore,
            path_on_datastore,
        ),
        validate=True,
        ).mount()
    trades_raw_dataset.start()
    files_list = os.listdir(trades_raw_dataset.mount_point)
    print("there are {} files in the input dataset".format(len(files_list)))
    merged_files = os.listdir(args.data_output)
    print("there are {} files in the merged dataset".format(len(merged_files)))
    merged_files = [x.replace('-mt.csv','.csv.gz').replace('-mts.csv','.csv.gz') for x in merged_files]
    remaining_files= set(files_list).difference(set(merged_files))
    remaining_files = list(remaining_files)
    print("there are {} files remaining in the input dataset".format(len(remaining_files)))
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(run, [[os.path.join(trades_raw_dataset.mount_point,name)] for name in remaining_files])
    # for i in tqdm(range(len(remaining_files))):
    #     if '.csv.gz' in remaining_files[i]:
    #         results = run([os.path.join(trades_raw_dataset.mount_point,remaining_files[i])])
    trades_raw_dataset.stop()
    with open("trade_merge_logs_sample.log",'w') as f:
        f.writelines([str(x)+'\n' for x in results])
        