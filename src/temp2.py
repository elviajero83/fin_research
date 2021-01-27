import pandas as pd
from misc_package import pretty_print
import datetime
import numpy as np
import os
import sqlite3 as lite
from multiprocessing import Process
from date_functions import date_ranges
import time
import sys
import math

# Initialization

# Where to get raw data from
scratch_path = "/tr/liquidity_futures/data/"
new_scratch = "/tr/scratch-liquidity_futures/data/"
alternate_source_path = os.path.join(new_scratch, "raw_data", "daily")
source_path = os.path.join(new_scratch, "raw_data", "daily")

# Where the cleaned data will go
target_path = os.path.join(scratch_path, "clean_data")

RIC_list_file = "/tr/liquidity_futures/RIC_list_tick_by_tick.db"
#RIC_list_file = "/tr/proj14/futures_liquidity/RIC_list_tick_by_tick.db"

class data_process():
    # WARNING: For the outlier detection to work properly all the NaN values should be removed prior to calling the
    # function
    TRIM_DELTA = 10

    def trim_mean(self, df):
        if len(df) == 0:
            return np.NaN

        low_cut = np.percentile(df, self.TRIM_DELTA/2)
        high_cut = np.percentile(df, 100 - self.TRIM_DELTA/2)

        index1 = (df <= high_cut)
        index2 = (df >= low_cut)
        index = np.logical_and(index1, index2)

        if len(df[index]) == 0:
            return np.NaN

        return df[index].mean()

    def trim_std(self, df):
        if len(df) == 0:
            return np.NaN

        low_cut = np.percentile(df, self.TRIM_DELTA/2)
        high_cut = np.percentile(df, 100 - self.TRIM_DELTA/2)

        index1 = (df <= high_cut)
        index2 = (df >= low_cut)
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
    def detect_outliers(self, df, m, gamma_multiple, data_column = "Price"):
        # Length of the window
        k = 2 * m + 1

        if len(df.index) < k:
            df['Outlier'] = False
            return

        old_setting = np.seterr('raise')

        if df[data_column].isnull().values.any():
            raise Exception("Price column contains NaN values.")

        df['Trim Mean'] = df[data_column].rolling(window=k, min_periods=k, center=True).apply(self.trim_mean, raw=True)
        df['Trim Std'] = df[data_column].rolling(window=k, min_periods=k, center=True).apply(self.trim_std, raw=True)

        # Recalculate for the first m observations
        top = df.head(n=k).copy(deep=True)
        top['Trim Mean'] = self.trim_mean(top[data_column].values)
        top['Trim Std'] = self.trim_std(top[data_column].values)
        df[:m] = top

        # Recalculate for the last m observations
        bottom = df.tail(n=k).copy(deep=True)
        bottom['Trim Mean'] = self.trim_mean(bottom[data_column].values)
        bottom['Trim Std'] = self.trim_std(bottom[data_column].values)
        df[-m:] = bottom

        # This line throws a warning if all observations result in NaNs
        df['PRICE_CHANGE'] = np.absolute(df[data_column] - df[data_column].shift(1))
        df['PRICE_CHANGE'] = np.where(df['PRICE_CHANGE'] == 0, np.nan, df['PRICE_CHANGE'])
        MIN_PRICE_CHANGE = df['PRICE_CHANGE'].min()
        gamma = gamma_multiple * MIN_PRICE_CHANGE
        
        # Changing gamma if all of the prices in a day are the same (price change = nan)
        if math.isnan(gamma): 
            gamma = 1e6
            
        df['Outlier'] = ~(np.abs(df[data_column] - df['Trim Mean']) < 3 * df['Trim Std'] + gamma)
        df.drop('PRICE_CHANGE', 1, inplace=True)
        df.drop('Trim Mean', 1, inplace=True)
        df.drop('Trim Std', 1, inplace=True)

        np.seterr(**old_setting)


# This function is invoked to print the time every 1 second
def print_status(pretty_printer, start_time):
    while (True):
        elapsed_time = time.time() - start_time
        pretty_printer.print_elapsed_time(elapsed_time)
        time.sleep(1)

# Returns the the last date consecutive date available
def save_last_date(dt_start_date, completed_list, data_types):

    sorted_list = sorted(completed_list, key=lambda x: (x['aggregation_frequency'], x['str_date']))

    overall_safe_date = datetime.datetime.now()
    for data_type in data_types:
        my_list = [item for item in sorted_list if item['aggregation_frequency'] == data_type]

        # Last Safe date for Quotes
        current_date = dt_start_date
        for item in my_list:
            dt_list_date = datetime.datetime.strptime(item['str_date'], "%Y-%m-%d")
            if (dt_list_date == current_date):
                current_date = current_date + datetime.timedelta(days=1)
            else:
                break

        my_list_safe_date = current_date + datetime.timedelta(days=-1)
        if my_list_safe_date < overall_safe_date:
            overall_safe_date = my_list_safe_date

    return overall_safe_date.strftime("%Y-%m-%d")


# infer trade direction
# This is based on Lee and Ready (1991)
def trade_direction(row):
    if (row['Ask Price'] == None or row['Bid Price'] == None or row['Ask Price'] < row['Bid Price']):
        return 0
    else:
        mid_quote = (row['Ask Price'] + row['Bid Price']) / 2
        if row['Price'] > mid_quote:
            # Sell
            return -1
        elif (row['Price'] < mid_quote):
            # Buy
            return 1
        else:
            # Tick test
            if (row['Lag(Price) Distinct'] == None):
                return 0
            else:
                if (row['Price'] > row['Lag(Price) Distinct']):
                    # Buy
                    return 1
                else:
                    # Sell
                    return -1

# This function processes a single day tick data and saves the aggregated version in the target folder
def aggregate_single_ric_single_day(ric_base, str_date, input_path, output_path, aggregation_frequency):
    try:
        full_quote_clean_file_name = os.path.join(input_path, ric_base, "Quotes", "corrected", "cleaned_%s.csv.gz" % (str_date))
        full_trade_clean_file_name = os.path.join(input_path, ric_base, "Trades", "corrected", "cleaned_%s.csv.gz" % (str_date))

        # Use clean files if they exist
        if os.path.exists(full_quote_clean_file_name):
            full_quote_file_name = full_quote_clean_file_name
        else:
            full_quote_file_name = os.path.join(input_path, ric_base, "Quotes", "corrected", "%s.csv.gz" % (str_date))

        if os.path.exists(full_trade_clean_file_name):
            full_trade_file_name = full_trade_clean_file_name
        else:
            full_trade_file_name = os.path.join(input_path, ric_base, "Trades", "corrected", "%s.csv.gz" % (str_date))

        full_target_file_name = os.path.join(output_path, str_date + ".csv")

        # Read the Quote and Trade data
        dtype_quote = {
            '#RIC': object,
            'Domain': object,
            'Date-Time': object,
            'Type': object,
            'Bid Price': float,
            'Ask Price': float,
            'Ask Size': float,
            'Seq. No.': object,
            'Exch Time': object,
        }
        df_quote_all_RICs = pd.read_csv(full_quote_file_name, compression='gzip', sep=',', dtype=dtype_quote)

        dtype_trade = {
            '#RIC': object,
            'Domain': object,
            'Date-Time': object,
            'Type': object,
            'Price': float,
            'Volume': float,
            'Seq. No.': object,
            'Exch Time': object,
        }
        df_trade_all_RICs = pd.read_csv(full_trade_file_name, compression='gzip', sep=',', dtype=dtype_trade)
   
   
    except Exception as e:
        if (e.__class__.__name__ == 'EmptyDataError'):
            sys.exit(1)
            # The file is empty most likely for a holiday. No need to process. Return Success.
        
        # Return fail as some other error happened.
        sys.exit (-1)

    try:
        #print(df_trade_all_RICs[["#RIC", 'Volume']].groupby('#RIC').sum())

        RIC_list = df_quote_all_RICs['#RIC'].unique()

        first_RIC = True
        for RIC in RIC_list:
            # Create separate data for quotes and trades of the specific RIC
            df_quote = df_quote_all_RICs.loc[df_quote_all_RICs['#RIC'] == RIC, :].copy(deep=True)
            df_trade = df_trade_all_RICs.loc[df_trade_all_RICs['#RIC'] == RIC, :].copy(deep=True)

            # Convert to time series
            df_quote['datetime'] = pd.to_datetime(df_quote['Date-Time'].str[0:26], format='%Y-%m-%dT%H:%M:%S.%f')
            df_quote = df_quote.set_index(pd.DatetimeIndex(df_quote['datetime']))

            df_trade['datetime'] = pd.to_datetime(df_trade['Date-Time'].str[0:26], format='%Y-%m-%dT%H:%M:%S.%f')
            #df_trade = df_trade.set_index(pd.DatetimeIndex(df_trade['datetime']))
            df_trade = df_trade.reset_index()

            # Used to generate full time index instead of only ones with data
            dt_file_date_start = datetime.datetime.strptime(str_date, "%Y-%m-%d")
            dt_file_date_end = dt_file_date_start + datetime.timedelta(days=1, microseconds=-1)

            # Aggregation Freqquency
            rule = aggregation_frequency
            df_quote_agg = pd.DataFrame()

            # it seems TRTH doesn't do forward filling when calculating the aggregated version
            # # forward fill bid and ask
            # df_quote['Ask Price'].ffill(inplace=True)
            # df_quote['Ask Size'].ffill(inplace=True)
            # df_quote['Bid Price'].ffill(inplace=True)
            # df_quote['Bid Size'].ffill(inplace=True)

            # Clean Trade Data

            # Uncommment this one as it was a one time thing when CL price was negative on March 20th
            df_trade['Price'] = np.where(df_trade['Price'] <= 0, np.nan, df_trade['Price'])
            df_trade['Volume'] = np.where(df_trade['Volume'] <= 0, np.nan, df_trade['Volume'])
            df_trade['Price'] = np.where(df_trade['Volume'].isnull(), np.nan, df_trade['Price'])
            df_trade['Volume'] = np.where(df_trade['Price'].isnull(), np.nan, df_trade['Volume'])
            df_trade = df_trade.loc[~df_trade['Price'].isnull(), :].copy(deep=True)

            # # Take care of trades with duplicate time stamps
            # df_trade = df_trade.groupby(df_trade.index).agg({'#RIC': 'first',
            #                                                  'Price': 'mean',
            #                                                  'Volume': 'sum',
            #                                                  'td_count': 'count'})
            #


            # Detect Outliers in Trade Prices
            cleaner = data_process()
            cleaner.detect_outliers(df_trade, 40, 6, 'Price')

            df_trade['Price'] = np.where(df_trade['Outlier'], np.nan, df_trade['Price'])
            df_trade['Volume'] = np.where(df_trade['Outlier'], np.nan, df_trade['Volume'])

            # duplicate index causes problems with outlier detection, and hence index was set to row number
            # now that the outliers are removed, it is safe to use datetime as index
            df_trade = df_trade.set_index(pd.DatetimeIndex(df_trade['datetime']))

            # Clean Quote Data
            df_quote['Ask Size'] = np.where(df_quote['Ask Size'] <= 0, np.nan, df_quote['Ask Size'])
            df_quote['Ask Price'] = np.where(df_quote['Ask Size'].isnull(), np.nan, df_quote['Ask Price'])
            # Uncommment this one as it was a one time thing when CL price was negative on March 20th
            # df_quote['Ask Price']= np.where(df_quote['Ask Price'] == 0, np.nan, df_quote['Ask Price'])
            df_quote['Ask Size'] = np.where(df_quote['Ask Price'].isnull(), np.nan, df_quote['Ask Size'])

            df_quote['Bid Size'] = np.where(df_quote['Bid Size'] <= 0, np.nan, df_quote['Bid Size'])
            df_quote['Bid Price'] = np.where(df_quote['Bid Size'].isnull(), np.nan, df_quote['Bid Price'])
            df_quote['Bid Price'] = np.where(df_quote['Bid Price'] == 0, np.nan, df_quote['Bid Price'])
            df_quote['Bid Size'] = np.where(df_quote['Bid Price'].isnull(), np.nan, df_quote['Bid Size'])

            # # Detect Outlier in Asks and Bids,
            # df_ask = df_quote[['datetime', 'Ask Price', 'Ask Size']].loc[~df_quote['Ask Price'].isnull(), :].copy(deep=True)
            # df_bid = df_quote[['datetime', 'Bid Price', 'Bid Size']].loc[~df_quote['Bid Price'].isnull(), :].copy(deep=True)
            #
            # cleaner = data_process()
            # cleaner.detect_outliers(df_ask, 40, 6, 'Ask Price')
            # df_ask['Ask Price'] = np.where(df_ask['Outlier'], np.nan, df_ask['Ask Price'])
            # df_ask['Ask Size'] = np.where(df_ask['Outlier'], np.nan, df_ask['Ask Size'])
            #
            # cleaner = data_process()
            # cleaner.detect_outliers(df_quote, 40, 6, 'Bid Price')
            # df_bid['Bid Price'] = np.where(df_bid['Outlier'], np.nan, df_bid['Bid Price'])
            # df_bid['Bid Size'] = np.where(df_bid['Outlier'], np.nan, df_bid['Bid Size'])
            #
            # print (df_ask.head(5), df_bid.head(5))
            # df_quote = pd.concat(df_bid, df_ask)
            # print (df_quote.head(20))
            # time.sleep(1000)

            # calculate the aggregate quote variables
            df_quote_agg['Open Ask'] = df_quote['Ask Price'].resample(rule=rule).first()
            df_quote_agg['Open Ask Size'] = df_quote['Ask Size'].resample(rule=rule).first()

            df_quote_agg['Close Ask'] = df_quote['Ask Price'].resample(rule=rule).last()
            df_quote_agg['Close Ask Size'] = df_quote['Ask Size'].resample(rule=rule).last()

            df_quote_agg['High Ask'] = df_quote['Ask Price'].resample(rule=rule).max()
            df_quote_agg['Low Ask'] = df_quote['Ask Price'].resample(rule=rule).min()
            df_quote_agg['Avg Ask Size'] = df_quote['Ask Size'].resample(rule=rule).mean()

            df_quote_agg['Open Bid'] = df_quote['Bid Price'].resample(rule=rule).first()
            df_quote_agg['Open Bid Size'] = df_quote['Bid Size'].resample(rule).first()

            df_quote_agg['Close Bid'] = df_quote['Bid Price'].resample(rule=rule).last()
            df_quote_agg['Close Bid Size'] = df_quote['Bid Size'].resample(rule).last()

            df_quote_agg['High Bid'] = df_quote['Bid Price'].resample(rule=rule).max()
            df_quote_agg['Low Bid'] = df_quote['Bid Price'].resample(rule=rule).min()
            df_quote_agg['Avg Bid Size'] = df_quote['Bid Size'].resample(rule=rule).mean()


            df_quote_agg['Dollar Spread'] = df_quote_agg['Close Ask'] - df_quote_agg['Close Bid']
            df_quote_agg['Pct Spread'] = df_quote_agg['Dollar Spread'] / \
                                         (df_quote_agg['Close Ask'] + df_quote_agg['Close Bid']) * 2 * 10000
            df_quote_agg['Log Spread'] = (np.log(df_quote_agg['Close Ask']) - np.log(df_quote_agg['Close Bid'])) * 10000
            # Need to take care of cross quotes (Ask < Bid)
            df_quote_agg['Dollar Spread'] = np.where(df_quote_agg['Close Bid'] >= df_quote_agg['Close Ask'], np.nan,
                                                     df_quote_agg['Dollar Spread'])
            df_quote_agg['Pct Spread'] = np.where(df_quote_agg['Close Bid'] >= df_quote_agg['Close Ask'], np.nan,
                                                     df_quote_agg['Pct Spread'])
            df_quote_agg['Log Spread'] = np.where(df_quote_agg['Close Bid'] >= df_quote_agg['Close Ask'], np.nan,
                                                     df_quote_agg['Log Spread'])




            # calculate the aggregate trade variables
            df_trade_agg = pd.DataFrame()

                      
            df_trade['Price*Volume'] = df_trade['Price'] * df_trade['Volume']
            #df_trade['Price*Volume'] = df_trade['Price'] * float(df_eod['Volume'])

            try:
                df_trade_agg['Open Price'] = df_trade['Price'].resample(rule=rule).first()
                df_trade_agg['Close Price'] = df_trade['Price'].resample(rule=rule).last()
                df_trade_agg['High Price'] = df_trade['Price'].resample(rule=rule).max()
                df_trade_agg['Low Price'] = df_trade['Price'].resample(rule=rule).min()
                df_trade_agg['Avg Price'] = df_trade['Price'].resample(rule=rule).mean()

                df_trade_agg['VW Price'] = df_trade['Price*Volume'].resample(rule=rule).sum() / df_trade['Volume'].resample(rule=rule).sum()
                #df_trade_agg['VW Price'] = df_trade['Price*Volume'].resample(rule=rule).sum() / float(df_eod['Volume'])
                df_trade.drop('Price*Volume', 1)
                #df_trade_agg['EoD Volume'] = df_eod['Volume']
                df_trade_agg['Avg Trade Size'] = df_trade['Volume'].resample(rule=rule).mean()
                df_trade_agg['Volume'] = df_trade['Volume'].resample(rule=rule).sum()
                df_trade_agg['Trade Count'] = df_trade['Volume'].resample(rule=rule).count()
                df_trade_agg['Return'] = (df_trade_agg['Close Price'] - df_trade_agg['Close Price'].shift(1)) / df_trade_agg[
                    'Close Price'].shift(1) * 10000
                df_trade_agg['Log Return'] = (np.log(df_trade_agg['Close Price']) - np.log(
                    df_trade_agg['Close Price'].shift(1))) * 10000



                if ric_base == 'ES': 
                    df_trade_agg['Notational Trade Size'] = df_trade_agg['Close Price'] * df_trade_agg['Volume'] * 50
                else:
                    # Will add in later for others
                    df_trade_agg['Notational Trade Size'] = np.NaN

                # Form lag(price) with distinct value
                df_trade['temp_lag_price'] = df_trade['Price'].shift(1)
                df_trade['Lag(Price) Distinct'] = np.where(df_trade['temp_lag_price'] == df_trade['Price'], np.NaN,
                                                           df_trade['temp_lag_price'])
                df_trade['Lag(Price) Distinct'].ffill(inplace=True)
                df_trade.drop('temp_lag_price', 1)

                # Merge Quotes and Trades
                df_quote['Type'] = 'Quote'
                df_all = pd.concat([df_quote[['Type', 'Ask Price', 'Bid Price', 'Seq. No.']],
                     df_trade[['Price', 'Lag(Price) Distinct', 'Volume', 'Seq. No.']]], sort=True).sort_values(by=['datetime', 'Seq. No.'])
                df_all['Type'].fillna(value='Trade', inplace=True)
                df_all['Ask Price'].ffill(inplace=True)
                df_all['Bid Price'].ffill(inplace=True)
                df_all = df_all[df_all['Type'] == 'Trade']

                # Mid Quote
                df_all['Mid Quote'] = (df_all['Ask Price'] + df_all['Bid Price']) / 2

                # Setting Tick Dir
                # Buy
                df_all['case1'] = np.where(df_all['Price'] > df_all['Mid Quote'], 1, 0)
                # Sell
                df_all['case2'] = np.where(df_all['Price'] < df_all['Mid Quote'], -1, 0)
                # Buy
                df_all['case3'] = np.where((df_all['Price'] == df_all['Mid Quote']) &
                                           (df_all['Price'] > df_all['Lag(Price) Distinct']), 1, 0)
                # Sell
                df_all['case4'] = np.where((df_all['Price'] == df_all['Mid Quote']) &
                                           (df_all['Price'] < df_all['Lag(Price) Distinct']), -1, 0)
                df_all['Tick Dir'] = df_all['case1'] + df_all['case2'] + df_all['case3'] + df_all['case4']
                df_all['Signed Trade SQRT'] = df_all['Tick Dir'] * np.sqrt(df_all['Volume'])
                df_trade_agg['Agg Trade SQRT'] = df_all['Signed Trade SQRT'].resample(rule=rule).sum()
                df_all['Signed Trade'] = df_all['Tick Dir'] * (df_all['Volume'])
                df_trade_agg['Agg Trade'] = df_all['Signed Trade'].resample(rule=rule).sum()
            except Exception as e:
                # No trades reproted
                if not df_trade.empty:
                    raise("There are trades but resampling failed")
                df_trade_agg['Open Price'] = np.nan
                df_trade_agg['Close Price'] = np.nan
                df_trade_agg['High Price'] = np.nan
                df_trade_agg['Low Price'] = np.nan
                df_trade_agg['Avg Price'] = np.nan
                df_trade_agg['VW Price'] = np.nan
                df_trade_agg['Avg Trade Size'] = np.nan
                df_trade_agg['Volume'] = np.nan
                df_trade_agg['Trade Count'] = np.nan
                df_trade_agg['Return'] = np.nan
                df_trade_agg['Log Return'] = np.nan
                df_trade_agg['Agg Trade'] = np.nan
                df_trade_agg['Agg Trade SQRT'] = np.nan
                df_trade_agg['Notational Trade Size'] = np.nan
                #df_trade_agg['EoD Volume'] = np.nan
                # print ("There was an error ")
                # print (e)

            my_index = pd.date_range(start=dt_file_date_start, end=dt_file_date_end, freq=rule, closed='left')
            df_agg = pd.DataFrame(index=my_index)

            df_agg['RIC'] = RIC
            df_agg['Date[L]'] = df_agg.index.strftime("%Y-%m-%d")
            df_agg['Time[L]'] = df_agg.index.strftime("%H:%M:%S")


            df_agg = pd.concat([df_trade_agg, df_quote_agg, df_agg], axis=1)

            if (first_RIC):
                write_mode = "w"
            else:
                write_mode = 'a'

            df_agg[['Date[L]', 'Time[L]', 'RIC', 'Open Price', 'Close Price', 'Low Price', 'High Price',
                    'Avg Price', 'VW Price', 'Trade Count', 'Volume', 'Return', 'Log Return', 'Agg Trade', 'Agg Trade SQRT',
                    'Avg Trade Size', 'Notational Trade Size',
                    'Open Ask', 'Open Ask Size', 'High Ask', 'Low Ask', 'Close Ask', 'Close Ask Size', 'Avg Ask Size',
                    'Open Bid', 'Open Bid Size', 'High Bid', 'Low Bid', 'Close Bid', 'Close Bid Size', 'Avg Bid Size',
                    'Dollar Spread', 'Pct Spread', 'Log Spread']] \
                .to_csv(full_target_file_name, mode=write_mode, header=first_RIC,
                        float_format="%.4f", index=False, index_label="Datetime")

            first_RIC = False

        # Exit the function successfully
        sys.exit(1)
    except Exception as e:
        print (e)
        sys.exit(-1)

def aggregate_single_ric(ric_base, dt_start_date, dt_end_date, aggregation_frquency):

    dt_current_date = dt_start_date
    aggregate_path = os.path.join(target_path, ric_base, aggregation_frquency) 
    
    # Create the target path if it doesn't exist
    if not os.path.exists(aggregate_path):
        os.makedirs(aggregate_path)

    while (dt_current_date <= dt_end_date):
        dt_current_date = dt_current_date + datetime.timedelta(days=1)
        str_date = dt_current_date.strftime("%Y-%m-%d")

        quote_file = os.path.join(source_path, ric_base, "Quotes", "corrected", "%s.csv.gz" % (str_date) )
        trade_file = os.path.join(source_path, ric_base, "Trades", "corrected", "%s.csv.gz" % (str_date) )
        # check if the files exist
        if not os.path.isfile(quote_file):
            # Check in the alternate source folder
            quote_file = os.path.join(alternate_source_path, ric_base, "Quotes", "corrected", "%s.csv.gz" % (str_date) )
            if not os.path.isfile(quote_file):
                continue
        if not os.path.isfile(trade_file):
            # Check in the alternate source folder
            trade_file = os.path.join(source_path, ric_base, "Trades", "corrected", "%s.csv.gz" % (str_date))
            if not os.path.isfile(trade_file):
                continue

        print ("Processing %s(%s) %s" %(ric_base, aggregation_frquency, str_date))
        aggregate_single_ric_single_day(ric_base, str_date, source_path, aggregate_path, aggregation_frquency)


def aggregate_single_ric_parallel (ric_base, dt_start_date, dt_end_date, adhoc=False):
    NUMBER_OF_PROCESS = 30

    agg_freq_list = ['1T', '5T']
    #if ric_base == '1S1R' or ric_base == '1SRA':
    if ric_base == 'FF':
        agg_freq_list =['1T', '5T', '30T', '60T']

    my_printer = pretty_print.pretty_batch_status(NUMBER_OF_PROCESS,
                                                  ric_base,
                                                  dt_start_date.strftime("%Y-%m-%d"),
                                                  dt_end_date.strftime("%Y-%m-%d"),
                                                  str_title = "Welcome to tick data aggregator by Erfan Danesh",
                                                  log_print=True)
    my_printer.print_safe_date((dt_start_date + datetime.timedelta(days=-1)).strftime("%Y-%m-%d"))

    my_date_range = date_ranges.break_into_multiday(dt_start_date, dt_end_date, 1)

    tasks_queue = []
    completed_tasks = []

    for time_range in my_date_range:
        str_date = time_range[0].strftime("%Y-%m-%d")

        for aggregation_frequency in agg_freq_list:
            quote_file = os.path.join(source_path, ric_base, "Quotes", "corrected", "%s.csv.gz" % (str_date) )
            trade_file = os.path.join(source_path, ric_base, "Trades", "corrected", "%s.csv.gz" % (str_date) )

            aggregate_path = os.path.join(target_path, ric_base, aggregation_frequency)        
            # Create the target path if it doesn't exist
            if not os.path.exists(aggregate_path):
                os.makedirs(aggregate_path)

            # check if the files exist
            input_path = source_path
            if (not os.path.isfile(quote_file)) or (not os.path.isfile(trade_file)):
                # File don't exist in the main source folder, check alternate source path
                quote_file = os.path.join(alternate_source_path, ric_base, "Quotes", "corrected", "%s.csv.gz" % (str_date))
                trade_file = os.path.join(alternate_source_path, ric_base, "Trades", "corrected", "%s.csv.gz" % (str_date))
                input_path = alternate_source_path
                if (not os.path.isfile(quote_file)) or (not os.path.isfile(trade_file)):
                    # The files don't exist, no need to process this day
                    # Add to the completed list
                    completed_tasks.append({'RIC': ric_base,
                                    'str_date': time_range[0].strftime("%Y-%m-%d"),
                                    'input_path': source_path,
                                    'output_path': aggregate_path,
                                    'aggregation_frequency': aggregation_frequency,
                                    })
                    continue

            tasks_queue.append({'RIC': ric_base,
                                'str_date': time_range[0].strftime("%Y-%m-%d"),
                                'input_path': input_path,
                                'output_path': aggregate_path,
                                'aggregation_frequency': aggregation_frequency,
                                })

    NUMBER_OF_HOLIDAY_TASKS = len(completed_tasks)
    NUMBER_OF_TASKS = len(tasks_queue)
    if NUMBER_OF_TASKS == 0:
        return

    ric_start_time = time.time()
    # Initiate Timer
    status_process = Process(target=print_status, args=(my_printer, ric_start_time))
    status_process.start()

    running_tasks = [None] * NUMBER_OF_PROCESS
    running_flags = [True] * NUMBER_OF_PROCESS
    process_array = [None] * NUMBER_OF_PROCESS
    tasks_timer = [None] * NUMBER_OF_PROCESS
    tasks_last_status_print = [None] * NUMBER_OF_PROCESS

    current_task_counter = 0

    while (any(running_flags)):
        for i in range(NUMBER_OF_PROCESS):
            if running_flags[i] == False:
                # This task is not running as there are no more jobs
                continue
            else:
                if running_tasks[i] == None:
                    # No task assigned. Look for new task
                    if (current_task_counter < NUMBER_OF_TASKS):
                        # There are more tasks available
                        tasks_timer[i] = time.time()
                        tasks_last_status_print[i] = tasks_timer[i]

                        running_tasks[i] = tasks_queue[current_task_counter]
                        current_task_counter +=1

                        process_array[i] = Process(target=aggregate_single_ric_single_day , args=(running_tasks[i]['RIC'],
                                                                                                  running_tasks[i]['str_date'],
                                                                                                  running_tasks[i]['input_path'],
                                                                                                  running_tasks[i]['output_path'],
                                                                                                  running_tasks[i]['aggregation_frequency'],
                                                                                                  ))
                        process_array[i].start()
                        my_printer.print_connection_status(i, "Aggregate   %s (%s) %7.2fs"
                                                                % (running_tasks[i]['str_date'],
                                                                   running_tasks[i]['aggregation_frequency'],
                                                                   time.time() - tasks_timer[i]))

                    else:
                        # No more tasks to assign.
                        running_flags[i]= False
                else:
                    # There is a task already assigned. Check the status in the next step
                    if process_array[i].exitcode == None:
                        # The task is still running. Update the on screen status
                        if time.time() - tasks_last_status_print[i] > 5:
                            tasks_last_status_print[i] = time.time()
                            my_printer.print_connection_status(i, "Aggregating %s (%s) %7.2fs"
                                                               % (running_tasks[i]['str_date'],
                                                                  running_tasks[i]['aggregation_frequency'],
                                                                  time.time() - tasks_timer[i]))
                    elif process_array[i].exitcode == 1:
                        # Task finished successfully
                        my_printer.print_connection_status(i, "Finished    %s (%s) %7.2fs"
                                                                % (running_tasks[i]['str_date'],
                                                                   running_tasks[i]['aggregation_frequency'],
                                                                   time.time() - tasks_timer[i]))
                        # Update the progress
                        completed_tasks.append(running_tasks[i])
                        safe_date = save_last_date(dt_start_date, completed_tasks, agg_freq_list)

                        if adhoc == False: 
                            con = lite.connect(RIC_list_file)
                            con.execute(
                                "UPDATE ric_chain_REST SET last_aggregation_date = '" + safe_date + "'WHERE RIC = '" + ric_base + "'")
                            con.commit()
                            con.close()

                        ric_pct_completed = (len(completed_tasks) - NUMBER_OF_HOLIDAY_TASKS) / (NUMBER_OF_TASKS * 1.0) * 100
                        ric_elapsed_time = (time.time() - ric_start_time)
                        ric_remaining_time = round(
                            ric_elapsed_time * (100 - ric_pct_completed) / ric_pct_completed)
                        ric_finish_time = datetime.datetime.today() + datetime.timedelta(
                            seconds=ric_remaining_time)

                        my_printer.print_safe_date(safe_date)
                        my_printer.print_pct(ric_pct_completed)
                        my_printer.print_finish_time(ric_finish_time)

                        #time.sleep(0.1)
                        running_tasks[i] = None
                    elif process_array[i].exitcode == -1:
                        my_printer.print_connection_status(i, "Failed %s (%s) %7.2fs"
                                                                % (running_tasks[i]['str_date'],
                                                                   running_tasks[i]['aggregation_frequency'],
                                                                   time.time() - tasks_timer[i]))
                        time.sleep(10)
                        # Rerun the task
                        process_array[i] = Process(target=aggregate_single_ric_single_day , args=(running_tasks[i]['RIC'],
                                                                                                  running_tasks[i]['str_date'],
                                                                                                  running_tasks[i]['input_path'],
                                                                                                  running_tasks[i]['output_path'],
                                                                                                  running_tasks[i]['aggregation_frequency'],
                                                                                                  ))
                        process_array[i].start()
                        my_printer.print_connection_status(i, "Aggregating %s (%s) %7.2fs"
                                                                % (running_tasks[i]['str_date'],
                                                                   running_tasks[i]['aggregation_frequency'],
                                                                   time.time() - tasks_timer[i]))
    status_process.terminate()



def aggregate_all_ric():
    # Open up dataset
    con = lite.connect(RIC_list_file)
    con.row_factory = lite.Row
    rows = []
    for row in con.execute('SELECT *  FROM ric_chain_REST WHERE aggregate = 1;'):
        rows.append(row)
    con.close()

    for row in rows:
        RIC = row['RIC']
        if RIC == 'VX:VE': 
            continue

        print ("Processing: %s" %(RIC))

        # Determine last date downloaded
        if (row['last_date_downloaded'] == ""):
            # if it is the first time download this RIC
            last_date_downloaded = datetime.datetime.strptime(row['start_date'], "%Y-%m-%d") + datetime.timedelta(days=-1)
        else:
            last_date_downloaded = datetime.datetime.strptime(row['last_date_downloaded'], "%Y-%m-%d")

        # Determine last date already aggregated
        if (row['last_aggregation_date'] == ""):
            last_aggregated_date = datetime.datetime.strptime(row['start_date'], "%Y-%m-%d") + datetime.timedelta(days=-1)
        else:
            last_aggregated_date = datetime.datetime.strptime(row['last_aggregation_date'].replace("\n",""), "%Y-%m-%d")


        start_date = last_aggregated_date + datetime.timedelta(days=1)
        #if RIC == 'SPY.P':
        #    start_date = datetime.datetime.strptime('2009-07-13', '%Y-%m-%d')


        # if (RIC != 'CL'):
        #      continue
        # start_date = datetime.datetime.strptime('2018-08-14', '%Y-%m-%d')
        # last_date_downloaded = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d')

        # This function generates 1T, 5T aggregations
        aggregate_single_ric_parallel(RIC, start_date, last_date_downloaded)
        time.sleep(1)


if __name__ == '__main__':
    aggregate_all_ric()

#start_date = datetime.datetime.strptime('2012-07-05', '%Y-%m-%d')
#end_date = datetime.datetime.strptime('2012-07-05', '%Y-%m-%d')
#aggregate_single_ric_parallel('2SP', start_date, end_date, adhoc=True)

#start_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')
#end_date = datetime.datetime.strptime('2013-12-31', '%Y-%m-%d')
#aggregate_single_ric_parallel('JY', start_date, end_date, adhoc=True)

# Get full list of RIC 
#def get_ric_list(input_path, ric_base):
#    full_quote_file_path = os.path.join(input_path, ric_base, "Quotes", "corrected")
#    full_quote_file_list = os.listdir(full_quote_file_path)
#    RIC_list = pd.DataFrame()
#    for date in full_quote_file_list: 
#         dtype_quote = {
#            '#RIC': object,
#            'Domain': object,
#            'Date-Time': object,
#            'Type': object,
#            'Bid Price': float,
#            'Ask Price': float,
#            'Ask Size': float,
#            'Seq. No.': object,
#            'Exch Time': object,
#        }
#        df_quote_all_RICs = pd.read_csv(full_quote_file_name, compression='gzip', sep=',', dtype=dtype_quote)
#        df_date_RIC = df_quote_a


