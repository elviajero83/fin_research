import argparse
import os
import numpy as np
import pandas as pd
from utils import merge_simultanous_rows, save_to_blob, data_process
from azureml.core import Run, Experiment, Workspace, Datastore, Dataset


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
        "--merge_path",
        type=str,
        required=False,
        default='datasets/trades_merged',
        help="path to merged files in Data Store",
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
    # df_har_data = pd.read_csv(path_to_har)
    run = Run.get_context()

    if "_OfflineRun" in str(run):
        ws = Workspace.from_config()
    else:
        ws = Run.get_context().experiment.workspace
    default_datastore = ws.get_default_datastore()


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []

    for raw_file_name in mini_batch:
        # read each file
        print("******  Processing {}".format(os.path.basename(raw_file_name)))
        df_1day_raw = pd.read_csv(raw_file_name, compression='gzip')
        print(df_1day_raw.shape)
        if df_1day_raw.shape[0] > 0:
            df_1day_raw = df_1day_raw.loc[:10000, :]

        print("shape of raw data: {}".format(df_1day_raw.shape))
        # Clean Trade Data
        df_1day_raw['Price'] = np.where(
            df_1day_raw['Price'] <= 0, np.nan, df_1day_raw['Price'])
        df_1day_raw['Volume'] = np.where(
            df_1day_raw['Volume'] <= 0, np.nan, df_1day_raw['Volume'])
        df_1day_raw['Price'] = np.where(
            df_1day_raw['Volume'].isnull(), np.nan, df_1day_raw['Price'])
        df_1day_raw['Volume'] = np.where(
            df_1day_raw['Price'].isnull(), np.nan, df_1day_raw['Volume'])
        df_1day_raw = df_1day_raw.loc[~df_1day_raw['Price'].isnull(), :].copy(
            deep=True)

        print("shape of data after removing negative  values: {}".format(
            df_1day_raw.shape))
        # # Take care of trades with duplicate time stamps
        # df_trade = df_trade.groupby(df_trade.index).agg({'#RIC': 'first',
        #                                                  'Price': 'mean',
        #                                                  'Volume': 'sum',
        #                                                  'td_count': 'count'})
        #

        # Detect Outliers in Trade Prices
        cleaner = data_process()
        cleaner.detect_outliers(df_1day_raw, 40, 6, 'Price')

        df_1day_raw['Price'] = np.where(
            df_1day_raw['Outlier'], np.nan, df_1day_raw['Price'])
        df_1day_raw['Volume'] = np.where(
            df_1day_raw['Outlier'], np.nan, df_1day_raw['Volume'])

        print("shape of data after removing outliers: {}".format(df_1day_raw.shape))
        # duplicate index causes problems with outlier detection, and hence index was set to row number
        # now that the outliers are removed, it is safe to use datetime as index
        # df_trade = df_trade.set_index(pd.DatetimeIndex(df_trade['datetime']))

        # clean unwanted cols
        df_1day_raw.drop(columns=[
                         '#RIC', 'Acc. Volume', 'Domain', 'Alias Underlying RIC', 'Type'], inplace=True)

        df_1day_raw.dropna(axis=0, inplace=True)
        df_1day_raw['Date-Time'] = pd.to_datetime(df_1day_raw['Date-Time'])
        df_1day_merged = df_1day_raw.groupby(
            'Date-Time').apply(merge_simultanous_rows)
        df_1day_merged["Acc Volume"] = df_1day_merged["Volume"].cumsum()
        print(df_1day_merged.shape)
        # df_1day_merged.to_csv(os.path.join('../data/test/', os.path.basename(raw_file_name))[:-3])
        save_to_blob(df=df_1day_merged, datastore=default_datastore, path=args.merge_path,
                     file_name=os.path.basename(raw_file_name).replace('.csv.gz', '.pkl'))
        # df_1day_merged.to_pickle('test.pkl')
        resultList.append("{}, {}".format(
            os.path.basename(raw_file_name), df_1day_merged.shape))

    return resultList


if __name__ == "__main__":

    init()
    print(args.file_path)
    run([args.file_path])
