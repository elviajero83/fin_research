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
        required=True,
        default='datasets/trades_merged',
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
    global df_har_data
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
    har_dataset = Dataset.File.from_files((default_datastore, "datasets/har"))
    with har_dataset.mount() as mount_context:
        df_har_data = pd.read_csv(select_first_file(mount_context.mount_point))
    df_har_data["Standard HAR (Log RV) 1-Month"] = df_har_data["HAR (Log RV) 1-Month"].pipe(scale)
    print("HAR df shape: {}".format(df_har_data.shape))


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []

    for raw_file_name in mini_batch:
        # read each file
        print("******  Processing {}".format(os.path.basename(raw_file_name)))
        df_1day_raw = pd.read_csv(raw_file_name, compression='gzip')
        print("original shape of data: {}".format(df_1day_raw.shape))
        if df_1day_raw.shape[0] == 0:
            resultList.append("{}, {}, {}".format(os.path.basename(raw_file_name), df_1day_raw.shape, 'None'))
            return resultList
        # get data volatality
        date_string = df_1day_raw.loc[0, 'Date-Time'][:10]
        print(date_string)
        daily_volatality_df = df_har_data[df_har_data["Date[L]"] == date_string][
            "Standard HAR (Log RV) 1-Month"
        ]
        if len(daily_volatality_df)>0: 
            daily_volatality = daily_volatality_df.values[0]
        else:
            print("No daily volatality data")
            resultList.append("{}, {}, {}".format(os.path.basename(raw_file_name), df_1day_raw.shape, 'None'))
            return resultList
        daily_volatality = 1 if np.isnan(
            daily_volatality) else daily_volatality
        print("daily volatility: {}".format(daily_volatality))
        df_1day_raw["dailyVolatility"] = daily_volatality

        # get right RIC
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

        # Remove unwanted cols
        drop_cols = ['Domain', 'Alias Underlying RIC', 'Type']
        df_1day_raw.drop(columns=drop_cols, inplace=True)

        # sample the data for faster process
        if args.sample and (df_1day_raw.shape[0] > 0):
            df_1day_raw = df_1day_raw.loc[:int(args.sample), :]

            print("shape of data after sampling: {}".format(df_1day_raw.shape))

        # Clean Trade Data- remove rows with negative dollar value
        cleaning_cols = ['Bid Price', 'Ask Price']
        df_1day_raw[cleaning_cols] = df_1day_raw[cleaning_cols].apply(
            lambda x: np.where(x <= 0, np.nan, x))
        df_1day_raw.dropna(axis=0, inplace=True, subset=cleaning_cols)

        print("shape of data after removing negative  values: {}".format(
            df_1day_raw.shape))

        df_1day_raw['Date-Time'] = pd.to_datetime(df_1day_raw['Date-Time'])
        df_1day_merged = df_1day_raw.groupby('Date-Time').agg({'Bid Price': 'max',
                                                               'Ask Price': 'min',
                                                               'Bid Size': 'sum',
                                                               'Ask Size': 'sum',
                                                               'Seq. No.': 'min',
                                                               'Exch Time': 'min',
                                                               '#RIC': 'min',
                                                               'dailyVolatility': 'min'})
        print("shape of data after merging simultanous quotes: {}".format(df_1day_merged.shape))

        # setting index back to numbers
        df_1day_merged.reset_index(level=0, inplace=True)

        # Save the merged data
        # save_to_blob(df=df_1day_merged, datastore=default_datastore, path=args.merge_path,
        #              file_name=os.path.basename(raw_file_name).replace('.csv.gz', '-mqs.pkl' if args.sample else '-mq.pkl'))
        print("creating folder {}".format(os.path.dirname(args.data_output)))
        os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
        file_name=os.path.basename(raw_file_name).replace('.csv.gz', '-mts.csv' if args.sample else '-mt.csv')
        df_1day_merged.to_csv(os.path.join(args.data_output, file_name), index=False)

        resultList.append("{}, {}, {}".format(
            os.path.basename(raw_file_name), df_1day_merged.shape, today_esv))


    return resultList


if __name__ == "__main__":
    init()
    print(args.file_path)
    run([args.file_path])