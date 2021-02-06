import sys
import pandas
import numpy
import requests
import os
import pandas as pd 
import numpy as np 
import datetime
from datetime import timezone
import utils

pd.options.display.max_rows = 999

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

# df_quotes_raw = pd.read_csv(quote_file, compression='gzip', nrows=50000)
# df_trades_raw = pd.read_csv(trade_file, compression='gzip', nrows=50000)

df_quotes_raw = pd.read_csv(quote_file, compression='gzip')
df_trades_raw = pd.read_csv(trade_file, compression='gzip')


df_quotes_raw = df_quotes_raw.loc[df_quotes_raw['#RIC'] == 'ESM9'].copy(deep=True).reset_index(drop=True)
df_trades_raw = df_trades_raw.loc[df_trades_raw['#RIC'] == 'ESM9'].copy(deep=True).reset_index(drop=True)

df_quotes_raw['Date-Time'] = pd.to_datetime(df_quotes_raw['Date-Time'])
df_trades_raw['Date-Time'] = pd.to_datetime(df_trades_raw['Date-Time'])

# Clean Quotes and Trades
cleaner = utils.data_process()

# Quotes
cleaner.set_negatives_to_nan(df_quotes_raw, keys=['Ask Price', 'Bid Price', 'Ask Size', 'Bid Size'])
cleaner.drop_nans(df_quotes_raw, keys=['Ask Price', 'Bid Price', 'Ask Size', 'Bid Size'])

# Trades
# Drop zero/negative 
cleaner.set_negatives_to_nan(df_trades_raw, keys=['Price', 'Volume'])
cleaner.drop_nans(df_trades_raw, keys=['Price', 'Volume'])

# Detect outliers
cleaner.detect_outliers(df_trades_raw, 40, 6, 'Price')
df_trades_raw['Price'] = np.where(df_trades_raw['Outlier'], np.nan, df_trades_raw['Price'])
df_trades_raw['Volume'] = np.where(df_trades_raw['Outlier'], np.nan, df_trades_raw['Volume'])
cleaner.set_negatives_to_nan(df_trades_raw, keys=['Price', 'Volume'])
cleaner.drop_nans(df_trades_raw, keys=['Price', 'Volume'])

# Forward fill quotes
for key in keys:
    df_quotes_raw[key] = df_quotes_raw[key].fillna(method='ffill')


# Form lag(price) with distinct value
df_trades_raw['temp_lag_price'] = df_trades_raw['Price'].shift(1)
df_trades_raw['Lag(Price) Distinct'] = np.where(df_trades_raw['temp_lag_price'] == df_trades_raw['Price'], np.NaN,
                                            df_trades_raw['temp_lag_price'])
df_trades_raw['Lag(Price) Distinct'].ffill(inplace=True)
df_trades_raw.drop('temp_lag_price', 1, inplace=True)

# Merge Quotes and Trades
df_quotes_raw['Type'] = 'Quote'
df_all = pd.concat([df_quotes_raw[['Date-Time', 'Type', 'Ask Price', 'Bid Price', 'Seq. No.']],
        df_trades_raw[['Date-Time', 'Price', 'Lag(Price) Distinct', 'Volume', 'Seq. No.']]], sort=True).sort_values(by=['Date-Time', 'Seq. No.'])

print (df_all.tail(300))

df_all['Type'].fillna(value='Trade', inplace=True)
df_all['Ask Price'].ffill(inplace=True)
df_all['Bid Price'].ffill(inplace=True)
df_all = df_all[df_all['Type'] == 'Trade']

# Mid Quote
df_all['Mid Quote'] = (df_all['Ask Price'] + df_all['Bid Price']) / 2

# Finding Tick Dir (Lee and Ready)
# Buy
df_all['case1'] = np.where(df_all['Price'] > df_all['Mid Quote'], 1, 0)
# Sell
df_all['case2'] = np.where(df_all['Price'] < df_all['Mid Quote'], -1, 0)
# Buy
df_all['case3'] = np.where((df_all['Price'] == df_all['Mid Quote']) & (df_all['Price'] > df_all['Lag(Price) Distinct']), 1, 0)
# Sell
df_all['case4'] = np.where((df_all['Price'] == df_all['Mid Quote']) & (df_all['Price'] < df_all['Lag(Price) Distinct']), -1, 0)

df_all['Tick Dir'] = df_all['case1'] + df_all['case2'] + df_all['case3'] + df_all['case4']
df_all['Signed Trade SQRT'] = df_all['Tick Dir'] * np.sqrt(df_all['Volume'])
df_all['Signed Trade'] = df_all['Tick Dir'] * (df_all['Volume'])

print (df_all.tail(30))


# Calculate some instantanous measures
df_quotes_raw['Spread'] = df_quotes_raw['Ask Price'] - df_quotes_raw['Bid Price']
df_quotes_raw['Mid Quote'] = (df_quotes_raw['Ask Price'] + df_quotes_raw['Bid Price']) / 2
df_quotes_raw['Smart Price'] = (df_quotes_raw['Ask Price']*(1/df_quotes_raw['Ask Size']) + df_quotes_raw['Bid Price']*(1/df_quotes_raw['Bid Size'])) / (1/df_quotes_raw['Ask Size'] + 1/df_quotes_raw['Bid Size'])
df_quotes_raw['Quote Imbalance'] = np.log(df_quotes_raw['Ask Size']) - np.log(df_quotes_raw['Bid Size'])

keys = keys + ['Spread', 'Mid Quote', 'Smart Price', 'Quote Imbalance']

# Time weighted measures based on quotes
df_TW_avg = utils.TW_avg(df=df_quotes_raw, datetime_col='Date-Time', keys=keys, timestamp_cutoffs = t_indices, fillforward=False)


# Measures based on trades (Ali probably has the code and will need to match up with that)
# Ali to write sum 'Signed Trade' between two bars to get 'Order Imbalance'
# 'Close Price', 'Avg Trade Size', 'Net Sell/Buy Count'=Sum Tick Dir, 'Ratio Sell/Buy=Sum(Tick Dir == -1)/sum(Tick Dir == 1)

