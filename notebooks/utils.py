# +
import time, sys
from IPython.display import clear_output
import pandas as pd

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def merge_simultanous_rows(x):
    d = {}
    # d['#RIC'] = x['#RIC'].values[0]
    # d['Alias Underlying RIC'] = x['Alias Underlying RIC'].values[0]
    # d['Domain'] = x['Domain'].values[0]
    # d['Type'] = x['Type'].values[0]
    d['Price'] = (x['Price']*x['Volume']).sum()/x['Volume'].sum()
    print('{}, {}'.format(x['Price'],x['Volume']))
    d['Volume'] = x['Volume'].sum()
    d['Seq. No.'] = x['Seq. No.'].values[0] #TODO: Ask Erfan
    d['Exch Time'] = x['Exch Time'].values[0] #TODO: Ask Erfan
    # d['Acc. Volume'] = x['Acc. Volume'].max()
    return pd.Series(d, index=['Price','Volume','Seq. No.','Exch Time'])

standarize = lambda x: (x-x.mean()) / x.std()

normalize = lambda x: (x-x.min()) / (x.max()-x.min()) 

scale = lambda x: (x/ x.mean())

def compute_label(events, current_ind, pt_level, sl_level, wait_time):
    pt_price = events.loc[current_ind, 'Price']*(1+pt_level)
    sl_price = events.loc[current_ind, 'Price']*(1-sl_level)
    end_time = events.loc[current_ind,'Date-Time']+wait_time
    last_ind = events.index.max()
    end_ind = int(np.nanmin([events[events['Date-Time']>end_time].index.min(), last_ind]))
    prices = events.loc[current_ind:end_ind, 'Price']
    pt_ind = prices[prices>pt_price].index.min()
    sl_ind = prices[prices<sl_price].index.min()
    # print(current_ind, (pt_ind, sl_ind, end_ind))
    return (pt_ind, sl_ind, end_ind)

def setlabel(x):
    if np.nanmin(x)==x['pt_ind']:
        return 1
    elif np.nanmin(x)==x['sl_ind']:
        return -1
    else:
        return 0

def set_df_labels(df, pt_level, sl_level, wait_time, vol_tick):
    df['pt_ind']=np.nan
    df['sl_ind']=np.nan
    df['end_ind']=np.nan
    df_bars = df[df['Acc Volume']%vol_tick ==0]
    for i in range(len(df_bars)):
        print(i)
        df_bars.loc[i, ['pt_ind', 'sl_ind', 'end_ind']] =compute_label(df, i, pt_level, sl_level, wait_time)

    df_bars['label'] = df_bars.loc[:,['pt_ind','sl_ind','end_ind']].apply(setlabel, axis =1)
    
    return df_bars

