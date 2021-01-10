import sys
import pandas
import numpy
import requests
import os
import pandas as pd 
import numpy as np 
import datetime
from datetime import timezone

def TW_avg(df, datetime_col, keys, timestamp_cutoffs, fillforward=True):
    # Forward Fill    
    if fillforward:
        for key in keys:
            df[key] = df[key].fillna(method='ffill')
            df['L1-'+key] = df[key].shift(1)

    # Form the interval groups based on the timestamps provided. The code doesn't automatically create any intervals at the 
    # begining and end of data. If desired the intervals should be explicitly passed to the function.
    df['Group'] = pd.cut(df[datetime_col], timestamp_cutoffs)
    df['Group Open'] = pd.IntervalIndex(df['Group']).get_level_values(0).left
    df['Group Close'] = pd.IntervalIndex(df['Group']).get_level_values(0).right

    # Forward Deltas
    df['F Delta'] = (df['Date-Time'].shift(-1)-df['Date-Time']).dt.total_seconds()
    df['F Delta 2'] = (df['Group Close']-df[datetime_col]).dt.total_seconds()
    df['F Delta 3'] = np.where((df['F Delta'] < df['F Delta 2']) | (df['F Delta'].isna()), df['F Delta'], df['F Delta 2'])

    # Backward Deltas
    df['B Delta'] = df['F Delta'].shift(1)
    df['B Delta 2'] = (df[datetime_col] - df['Group Open']).dt.total_seconds()
    df['B Delta 3'] = np.where((df['B Delta'] < df['B Delta 2']) | (df['B Delta'].isna()), np.NaN, df['B Delta 2'])

    # Variable * Delta
    for key in keys:
        df[key + ' * Delta'] = df[key] * df['F Delta 3']
        df['L1-' + key + ' * Delta'] = df['L1-' + key] * df['B Delta 3']

    # Group dataframe based on cutoffs
    df_grouped =  df.groupby(df['Group'])

    # Emoty dataframe for aggregate measures
    df_agg = pd.DataFrame()

    # Open and Close of Variables
    for key in keys:
        df_agg[key + ' * Delta'] = df_grouped[key + ' * Delta'].sum()
        df_agg[key + ' * Delta Open'] = df_grouped['L1-' + key + ' * Delta'].sum()
        df_agg['Time Delta'] = df_grouped['F Delta 3'].sum() + df_grouped['B Delta 3'].sum()

    df_agg['Bar Open Time Stamp'] =  pd.IntervalIndex(df_agg.index.get_level_values(0)).left
    df_agg['Bar Close Time Stamp'] =  pd.IntervalIndex(df_agg.index.get_level_values(0)).right

    for key in keys:
        df_agg['TW Avg '+ key] = (df_agg[key + ' * Delta'] + df_agg[key + ' * Delta Open'] ) / df_agg['Time Delta']
 
    return_cols = ['Time Delta', 'Bar Open Time Stamp', 'Bar Close Time Stamp'] + ['TW Avg '+key for key in keys]

    return df_agg[return_cols]
 





# READ DATA
str_date = "2019-05-08"
file_path = "/Users/mkalantari/.dropbox-two/Dropbox/Futures_ML_Project/output/raw_data/daily/ES"

# Sample Time Stamp Cutoffs
t_beg = datetime.datetime.fromisoformat('2019-05-08 03:59:00.000000-05:00')
t1 = datetime.datetime.fromisoformat('2019-05-08 04:00:00.000000-05:00')
t2 = datetime.datetime.fromisoformat('2019-05-08 04:00:01.000000-05:00')
t3 = datetime.datetime.fromisoformat('2019-05-08 04:00:02.000000-05:00')
t4 = datetime.datetime.fromisoformat('2019-05-08 04:00:06.000000-05:00')
t5 = datetime.datetime.fromisoformat('2019-05-08 04:01:06.000000-05:00')
t6 = datetime.datetime.fromisoformat('2019-05-08 04:10:06.000000-05:00')
t_end = datetime.datetime.fromisoformat('2019-05-08 22:31:00.000000-05:00')
t_indices = [t_beg, t1, t2, t3, t4, t5, t6, t_end] 

keys = ['Ask Price', 'Ask Size', 'Bid Price', 'Bid Size']

quote_file = os.path.join(file_path, 'Quotes', 'corrected', str_date + ".csv.gz")
trade_file = os.path.join(file_path, 'Trades', 'corrected', str_date + ".csv.gz")

df_quotes_raw = pd.read_csv(quote_file, compression='gzip', nrows=50000)
df_trades_raw = pd.read_csv(trade_file, compression='gzip', nrows=50000)

df_quotes_raw = df_quotes_raw.loc[df_quotes_raw['#RIC'] == 'ESH0'].copy()
df_quotes_raw['Date-Time'] = pd.to_datetime(df_quotes_raw['Date-Time'])

df_TW_avg = TW_avg(df=df_quotes_raw, datetime_col='Date-Time', keys=keys, timestamp_cutoffs = t_indices, fillforward=True)
print (df_TW_avg.head(20))
