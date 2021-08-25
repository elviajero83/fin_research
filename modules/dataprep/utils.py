import pandas as pd
import logging
from azureml.core import Run, Experiment, Workspace, Datastore, Dataset
import os
import time
import numpy as np
from datetime import datetime, timedelta
import sys


class data_process:
    # WARNING: For the outlier detection to work properly all the NaN values should be removed prior to calling the
    # function
    TRIM_DELTA = 10

    def trim_mean(self, df):
        if len(df) == 0:
            return np.NaN

        low_cut = np.percentile(df, self.TRIM_DELTA / 2)
        high_cut = np.percentile(df, 100 - self.TRIM_DELTA / 2)

        index1 = df <= high_cut
        index2 = df >= low_cut
        index = np.logical_and(index1, index2)

        if len(df[index]) == 0:
            return np.NaN

        return df[index].mean()

    def trim_std(self, df):
        if len(df) == 0:
            return np.NaN

        low_cut = np.percentile(df, self.TRIM_DELTA / 2)
        high_cut = np.percentile(df, 100 - self.TRIM_DELTA / 2)

        index1 = df <= high_cut
        index2 = df >= low_cut
        index = np.logical_and(index1, index2)

        if len(df[index]) == 0:
            return np.NaN

        return df[index].std()

    # This function detects the outliers based on the test introduced in Brownlees and Gallo (2006)
    # Inputs:
    #   m: m is used to calculate k as k=2m+1. K is the length of the window
    #   gamma_multiple: This is used to calculate gamma variables defined in the paper.
    #       We first calculate the MIN_PRICE_VARIATION for the day, and calculate gamma as
    #       gamma = gamma_multiple * MIN_PRICE_VARIATION
    # Constants:
    #   delta: trim parameter set to 10% as in the paper
    def detect_outliers(self, df, m, gamma_multiple, data_column="Price"):
        # Length of the window
        k = 2 * m + 1

        if len(df.index) < k:
            df["Outlier"] = False
            return

        old_setting = np.seterr("raise")

        if df[data_column].isnull().values.any():
            raise Exception("Price column contains NaN values.")

        df["Trim Mean"] = (
            df[data_column]
            .rolling(window=k, min_periods=k, center=True)
            .apply(self.trim_mean, raw=True)
        )
        df["Trim Std"] = (
            df[data_column]
            .rolling(window=k, min_periods=k, center=True)
            .apply(self.trim_std, raw=True)
        )

        # Recalculate for the first m observations
        top = df.head(n=k).copy(deep=True)
        top["Trim Mean"] = self.trim_mean(top[data_column].values)
        top["Trim Std"] = self.trim_std(top[data_column].values)
        df[:m] = top[:m]

        # Recalculate for the last m observations
        bottom = df.tail(n=k).copy(deep=True)
        bottom["Trim Mean"] = self.trim_mean(bottom[data_column].values)
        bottom["Trim Std"] = self.trim_std(bottom[data_column].values)
        df[-m:] = bottom[-m:]

        # This line throws a warning if all observations result in NaNs
        df["PRICE_CHANGE"] = np.absolute(df[data_column] - df[data_column].shift(1))
        df["PRICE_CHANGE"] = np.where(df["PRICE_CHANGE"] == 0, np.nan, df["PRICE_CHANGE"])
        MIN_PRICE_CHANGE = df["PRICE_CHANGE"].min()
        gamma = gamma_multiple * MIN_PRICE_CHANGE

        df["Outlier"] = ~(np.abs(df[data_column] - df["Trim Mean"]) < 3 * df["Trim Std"] + gamma)
        df.drop("PRICE_CHANGE", 1, inplace=True)
        df.drop("Trim Mean", 1, inplace=True)
        df.drop("Trim Std", 1, inplace=True)

        np.seterr(**old_setting)

    # This function replaces any zero or negative values in columns in keys to nan
    def set_negatives_to_nan(self, df, keys):
        for key in keys:
            df[key] = np.where(df[key] <= 0, np.nan, df[key])

    # Drops a row if it finds a nan value in any of the key columns
    def drop_nans(self, df, keys):
        df["condition"] = 0
        for key in keys:
            df["condition"] = df["condition"] | df[key].isnull()

        df.drop(df[df["condition"] == 1].index, inplace=True)
        df.drop("condition", 1, inplace=True)
        df.reset_index()

    def remove_negatives(self, df, keys):
        self.set_negatives_to_nan(df, keys)
        self.drop_nans(df, keys)

    def remove_outliers(self, df, m, gamma_multiple, data_column="Price"):
        self.detect_outliers(df, m, gamma_multiple, data_column)
        df[data_column] = np.where(df["Outlier"], np.nan, df[data_column])
        self.drop_nans(df, [data_column])


def merge_simultanous_rows(x):
    d = {}
    d["Price"] = (x["Price"] * x["Volume"]).sum() / x["Volume"].sum()
    d["Volume"] = x["Volume"].sum()
    d["Seq. No."] = x["Seq. No."].values[0]  # TODO: Ask Erfan
    d["Exch Time"] = x["Exch Time"].values[0]  # TODO: Ask Erfan
    return pd.Series(d, index=["Price", "Volume", "Seq. No.", "Exch Time"])


def merge_simultanous_rows_quotes(x):
    d = {}
    d["Price"] = (x["Price"] * x["Volume"]).sum() / x["Volume"].sum()
    d["Volume"] = x["Volume"].sum()
    d["Seq. No."] = x["Seq. No."].values[0]  # TODO: Ask Erfan
    d["Exch Time"] = x["Exch Time"].values[0]  # TODO: Ask Erfan
    return pd.Series(d, index=["Price", "Volume", "Seq. No.", "Exch Time"])


def save_to_blob(df, datastore, path, file_name):
    print("file_name: {}".format(file_name))
    df.to_pickle(file_name)
    time.sleep(1)
    datastore.upload_files(
        files=[file_name],
        relative_root=None,
        target_path=path,
        overwrite=True,
        show_progress=True,
    )


def standarize(x):
    return (x - x.mean()) / x.std()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def scale(x):
    return x / x.mean()


# def compute_label_long(events, current_ind, pt_level, sl_level, wait_time):
#     volatility = events.loc[current_ind, "dailyVolatility"]
#     pt_price = events.loc[current_ind, "Bid Price"] * (1 + (pt_level * volatility))
#     sl_price = events.loc[current_ind, "Bid Price"] * (1 - (sl_level * volatility))
#     end_time = events.loc[current_ind, "Date-Time"] + wait_time
#     last_ind = events.index.max()
#     end_ind = int(
#         np.nanmin([events[events["Date-Time"] > end_time].index.min(), last_ind])
#     )
#     prices = events.loc[current_ind:end_ind, "Bid Price"]
#     pt_ind = prices[prices > pt_price].index.min()
#     sl_ind = prices[prices < sl_price].index.min()
#     # print(current_ind, (pt_ind, sl_ind, end_ind))
#     return (pt_ind, sl_ind, end_ind)


def compute_label(events, current_ind, pt_level, sl_level, wait_time):
    volatility = events.loc[current_ind, "dailyVolatility"]
    pt_price_long = events.loc[current_ind, "Ask Price"] * (1 + (pt_level * volatility))
    sl_price_long = events.loc[current_ind, "Ask Price"] * (1 - (sl_level * volatility))
    pt_price_short = events.loc[current_ind, "Bid Price"] * (1 - (pt_level * volatility))
    sl_price_short = events.loc[current_ind, "Bid Price"] * (1 + (sl_level * volatility))
    end_time = events.loc[current_ind, "Date-Time"] + wait_time
    last_ind = events.index.max()
    end_ind = int(np.nanmin([events[events["Date-Time"] > end_time].index.min(), last_ind]))
    prices_long = events.loc[current_ind:end_ind, "Bid Price"]
    prices_short = events.loc[current_ind:end_ind, "Ask Price"]
    pt_long_ind = prices_long[prices_long > pt_price_long].index.min()
    sl_long_ind = prices_long[prices_long < sl_price_long].index.min()
    pt_short_ind = prices_short[prices_short < pt_price_short].index.min()
    sl_short_ind = prices_short[prices_short > sl_price_short].index.min()
    short_label = setlabel((pt_short_ind, sl_short_ind, end_ind))
    short_return = (
        prices_short[current_ind] - prices_short[np.nanmin((pt_short_ind, sl_short_ind, end_ind))]
    )
    short_duration = (
        events.loc[np.nanmin((pt_short_ind, sl_short_ind, end_ind)), "Date-Time"]
        - events.loc[current_ind, "Date-Time"]
    )
    long_label = setlabel((pt_long_ind, sl_long_ind, end_ind))
    long_return = (
        prices_long[np.nanmin((pt_long_ind, sl_long_ind, end_ind))] - prices_long[current_ind]
    )
    long_duration = (
        events.loc[np.nanmin((pt_long_ind, sl_long_ind, end_ind)), "Date-Time"]
        - events.loc[current_ind, "Date-Time"]
    )
    return np.array(
        (
            long_label,
            long_return,
            long_duration,
            pt_long_ind,
            sl_long_ind,
            short_label,
            short_return,
            short_duration,
            pt_short_ind,
            sl_short_ind,
            end_ind,
        )
    )


def setlabel(x):
    if np.nanmin(x) == x[0]:
        return 1
    elif np.nanmin(x) == x[1]:
        return -1
    else:
        return 0


def set_df_labels(df, pt_level, sl_level, wait_time, inds):
    df["pt_long_ind"] = np.nan
    df["sl_long_ind"] = np.nan
    df["pt_short_ind"] = np.nan
    df["sl_short_ind"] = np.nan
    df["end_ind"] = np.nan
    df["long_label"] = np.nan
    df["short_label"] = np.nan
    df["long_return"] = np.nan
    df["short_return"] = np.nan
    df["long_duration"] = np.nan
    df["short_duration"] = np.nan
    df_bars = df.loc[inds].copy()
    # print(df_bars.columns)
    for i in range(len(df_bars)):
        df_bars.loc[
            inds[i],
            [
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
            ],
        ] = compute_label(df, inds[i], pt_level, sl_level, wait_time)
    df_bars.reset_index(inplace=True)
    df_bars.rename(columns={"index": "original_index"}, inplace=True)

    print(df_bars.columns)
    return df_bars


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder

    Args:
        path (str): path to directory or file to choose

    Raises:
        ValueError: error raised when there are multiple files in the directory

    Returns:
        str: full path of selected file
    """
    if os.path.isfile(path):
        # log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "Input is file, selecting {}".format(path))
        return path

    files = os.listdir(path)
    # log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "Found {} in {}".format(files, path))
    if len(files) != 1:
        raise ValueError("expected exactly one file in directory")
    # log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "Selecting {}".format(files[0]))
    return os.path.join(path, files[0])


def TW_avg(input_df, datetime_col, keys, timestamp_cutoffs, fillforward=True):
    df = input_df.copy(deep=True)
    # Forward Fill
    if fillforward:
        for key in keys:
            df[key] = df[key].fillna(method="ffill")
            df["L1-" + key] = df[key].shift(1)

    # Form the interval groups based on the timestamps provided. The code doesn't automatically create any intervals at the
    # begining and end of data. If desired the intervals should be explicitly passed to the function.
    df["Group"] = pd.cut(df[datetime_col], timestamp_cutoffs)
    df["Group Open"] = pd.IntervalIndex(df["Group"]).get_level_values(0).left
    df["Group Close"] = pd.IntervalIndex(df["Group"]).get_level_values(0).right

    # Forward Deltas
    df["F Delta"] = (df["Date-Time"].shift(-1) - df["Date-Time"]).dt.total_seconds()
    df["F Delta 2"] = (df["Group Close"] - df[datetime_col]).dt.total_seconds()
    df["F Delta 3"] = np.where(
        (df["F Delta"] < df["F Delta 2"]) | (df["F Delta"].isna()),
        df["F Delta"],
        df["F Delta 2"],
    )

    # Backward Deltas
    df["B Delta"] = df["F Delta"].shift(1)
    df["B Delta 2"] = (df[datetime_col] - df["Group Open"]).dt.total_seconds()
    df["B Delta 3"] = np.where(
        (df["B Delta"] < df["B Delta 2"]) | (df["B Delta"].isna()),
        np.NaN,
        df["B Delta 2"],
    )

    # Variable * Delta
    for key in keys:
        df[key + " * Delta"] = df[key] * df["F Delta 3"]
        df["L1-" + key + " * Delta"] = df["L1-" + key] * df["B Delta 3"]

    # Group dataframe based on cutoffs
    df_grouped = df.groupby(df["Group"])

    # Emoty dataframe for aggregate measures
    df_agg = pd.DataFrame()

    # Open and Close of Variables
    for key in keys:
        df_agg[key + " * Delta"] = df_grouped[key + " * Delta"].sum()
        df_agg[key + " * Delta Open"] = df_grouped["L1-" + key + " * Delta"].sum()
        df_agg["Time Delta"] = df_grouped["F Delta 3"].sum() + df_grouped["B Delta 3"].sum()

    df_agg["Bar Open Time Stamp"] = pd.IntervalIndex(df_agg.index.get_level_values(0)).left
    df_agg["Bar Close Time Stamp"] = pd.IntervalIndex(df_agg.index.get_level_values(0)).right

    for key in keys:
        df_agg["TW Avg " + key] = (
            df_agg[key + " * Delta"] + df_agg[key + " * Delta Open"]
        ) / df_agg["Time Delta"]

    return_cols = ["Time Delta", "Bar Open Time Stamp", "Bar Close Time Stamp"] + [
        "TW Avg " + key for key in keys
    ]
    df_agg.reset_index(inplace=True)
    return df_agg[return_cols]


def trim_df_to_time(df, start_str, end_str):
    df["Time"] = df.loc[:, "Date-Time"].apply(lambda x: str(x)[11:-10])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
    print("full df shape", df.shape)
    df = df[df["Time"] >= datetime.strptime(start_str, "%H:%M:%S.%f")]
    print(">8:30  df shape", df.shape)
    df = df[df["Time"] <= datetime.strptime(end_str, "%H:%M:%S.%f")]
    print("time-trimmed df shape", df.shape)
    df.reset_index(inplace=True, drop=True)

    df["Date-Time"] = pd.to_datetime(df["Date-Time"])
    return df


def safe_division(n, d):
    return n / d if d else sys.maxsize


def enrich_with_trades_features(df_quotes, quotes_keys, df_trades):
    """attach the new cols based on merged trades and quotes
       it exects trades has 'Price' and 'Acc Volume' and quotes
       has quotes_keys

    Args:
        df_quotes ([type]): [description]
        quotes_keys ([type]): [description]
        df_trades ([type]): [description]
    """
    # Forward fill quotes
    for key in quotes_keys:
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
    df_trades_all["Mid Quote"] = (df_trades_all["Ask Price"] + df_trades_all["Bid Price"]) / 2

    # Finding Tick Dir (Lee and Ready)
    # Buy
    df_trades_all["case1"] = np.where(df_trades_all["Price"] > df_trades_all["Mid Quote"], 1, 0)
    # Sell
    df_trades_all["case2"] = np.where(df_trades_all["Price"] < df_trades_all["Mid Quote"], -1, 0)
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
    df_trades_all["Signed Trade"] = df_trades_all["Tick Dir"] * (df_trades_all["Acc Volume"])
    print("df_trades_all:")
    print(df_trades_all.head(5))

    return df_trades_all
