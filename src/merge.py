import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

data_dir = 'data/'

s_args = sys.argv[1:]
if s_args:
    data_dir = s_args[0]

def strtodate(s):
    try:
        return datetime.strptime(s, '%d/%m/%Y')
    except ValueError as e:
        pass # let it pass

    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError as e:
        raise

train_csvpath = os.path.join(data_dir, 'train.csv')
train_data = pd.read_csv(train_csvpath,
    dtype={
        'Store': int,
        'DayOfWeek': int,
        'Sales': float,
        'Open': int,
        'Customers': int,
        'Promo': int,
        'StateHoliday': str,
        'SchoolHoliday': int,
    },
    parse_dates=['Date'],
    date_parser=strtodate)

print '\'train.csv\' loaded'

train_data.insert(len(train_data.columns),
    'StateHoliday_0',
    (train_data['StateHoliday'] == '0').astype(int))

train_data.insert(len(train_data.columns),
    'StateHoliday_a',
    (train_data['StateHoliday'] == 'a').astype(int))

train_data.insert(len(train_data.columns),
    'StateHoliday_b',
    (train_data['StateHoliday'] == 'b').astype(int))

train_data.insert(len(train_data.columns),
    'StateHoliday_c',
    (train_data['StateHoliday'] == 'c').astype(int))

train_data.insert(len(train_data.columns),
    'Weekends',
    (5 <= train_data['DayOfWeek']).astype(int))

train_data.insert(len(train_data.columns),
    'Weekdays',
    (train_data['DayOfWeek'] < 5).astype(int))

print train_data.iloc[:5]

store_csvpath = os.path.join(data_dir, 'store.csv')
store_data = pd.read_csv(store_csvpath,
    dtype={
        'Store': int,
        'StoreType': str,
        'Assortment': str,
        'CompetitionDistance': float,
        'CompetitionOpenSinceMonth': float,
        'CompetitionOpenSinceYear': float,
        'Promo2': int,
        'Promo2SinceWeek': float,
        'Promo2SinceYear': float,
        'PromoInterval': str
    })

print '\'store.csv\' loaded'

store_data[['CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']].fillna(-1., inplace=True)

store_data[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
    'Promo2SinceWeek', 'Promo2SinceYear']].astype(int, copy=False)
