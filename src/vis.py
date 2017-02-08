import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# matplotlib.style.use('ggplot')

# plt.rcParams["figure.figsize"] = (16, 9)

print 'Fetching important packages'

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

epoch = datetime.utcfromtimestamp(0)

print 'Get something set up'

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

# Sort ascendingly according to date and store number
train_data.sort_values(['Date', 'Store'],
    axis=0,
    inplace=True,
    ascending=True)

print '\'train.csv\' loaded into pandas'

print 'Modifying \'train.csv\' for easier processing'

train_data.insert(len(train_data.columns),
    'StateHoliday_0',
    (train_data['StateHoliday'] == '0').astype(int, copy=False))

train_data.insert(len(train_data.columns),
    'StateHoliday_a',
    (train_data['StateHoliday'] == 'a').astype(int, copy=False))

train_data.insert(len(train_data.columns),
    'StateHoliday_b',
    (train_data['StateHoliday'] == 'b').astype(int, copy=False))

train_data.insert(len(train_data.columns),
    'StateHoliday_c',
    (train_data['StateHoliday'] == 'c').astype(int, copy=False))

train_data.insert(len(train_data.columns),
    'Weekends',
    (5 <= train_data['DayOfWeek']).astype(int, copy=False))

train_data.insert(len(train_data.columns),
    'Weekdays',
    (train_data['DayOfWeek'] < 5).astype(int, copy=False))

print train_data.iloc[:5]

def get_sma(arr, n=30):
    sz  = arr.size

    # Not the best method
    SMA = np.cumsum(arr, axis=None, dtype=float)

    for i in range(min(n, sz)):
        SMA[i] /= (i + 1)

    for i in range(n, sz):
        SMA[i] = SMA[i - 1] + (arr.item(i) - arr.item(i - n)) / n

    return SMA

print 'get_sma() is used to calculate Simple Moving Average'

num_stores = 1115
for i in range(1, 2):
    filtered_all = train_data[train_data['Store'] == i][
        ['DayOfWeek', 'Date', 'Sales', 'Customers',
        'Promo', 'StateHoliday', 'SchoolHoliday', 'Open', 'Weekends', 'Weekdays',
        'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']]

    filtered_open = train_data[(train_data['Store'] == i) &
        (train_data['Open'] == 1)][['DayOfWeek', 'Date', 'Weekends',
        'Sales', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Weekdays',
        'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']]

    for j in range(0, 3):
        year = 2013 + j

        filtered_all_yrly = filtered_all[filtered_all['Date'].dt.year == year]
        filtered_open_yrly = filtered_open[filtered_open['Date'].dt.year == year]
        
        impt_feats_all = filtered_all_yrly[['Sales', 'Promo',
            'Customers', 'Open', 'Weekends', 'Weekdays', 'SchoolHoliday',
            'StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']]

        corrmat_labls = list(impt_feats_all)
        corrmat = impt_feats_all.corr()

        dates_all = filtered_all_yrly['Date'].as_matrix()
        dates_open = filtered_open_yrly['Date'].as_matrix()
        dates_promo = filtered_open_yrly.loc[filtered_open_yrly['Promo'] == 1, 'Date'].as_matrix()
        dates_sch = filtered_all_yrly.loc[filtered_all_yrly['SchoolHoliday'] == 1, 'Date'].as_matrix()
        
        wkdays = filtered_all_yrly.loc[filtered_all_yrly['Weekdays'] == 1, 'Date'].as_matrix()
        wkends = filtered_all_yrly.loc[filtered_all_yrly['Weekends'] == 1, 'Date'].as_matrix()

        promos_open = filtered_open_yrly['Promo'].as_matrix()
        sales_open = filtered_open_yrly['Sales'].as_matrix()
        customers_open = filtered_open_yrly['Customers'].as_matrix()
        SMA7 = get_sma(sales_open, n=7)
        SMA21 = get_sma(sales_open, n=21)

        y_min = 0
        y_max = sales_open.max()

        plt.figure(1)
        plt.subplot(1, 1, 1)
        plt.vlines(wkdays, 0., y_max/2., color='#cccccc', zorder=-1)
        plt.vlines(wkends, y_max/2., y_max, color='yellow', zorder=0)
        plt.plot(dates_open, sales_open, '-b', zorder=1)
        plt.plot(dates_open, SMA21, '-r', zorder=2)
        plt.plot(dates_open, SMA7, '-g', zorder=3)
        plt.plot(dates_open, customers_open, '-c', zorder=4)
        plt.xlim((dates_all[0], dates_all[-1]))
        plt.ylim((y_min, y_max))
        plt.subplots_adjust(left=0.025, bottom=0., right=1., top=0.975, wspace=0., hspace=0.)
        
        plt.figure(2)
        plt.subplot(1, 1, 1)
        plt.vlines(dates_promo, y_max/2., y_max, color='yellow')
        plt.vlines(dates_sch, 0., y_max/2., color='#333333')
        plt.plot(dates_open, sales_open, '-b', zorder=3)
        plt.plot(dates_open, customers_open, '-c', zorder=4)
        plt.xlim((dates_all[0], dates_all[-1]))
        plt.ylim((y_min, y_max))
        plt.subplots_adjust(left=0.025, bottom=0., right=1., top=0.975, wspace=0., hspace=0.)
        
        fig = plt.figure(3)
        ax  = fig.add_subplot(1, 1, 1)
        cax = ax.matshow(corrmat, cmap='bwr', vmin=-1., vmax=1.)
        plt.xticks(range(len(corrmat_labls)), corrmat_labls, rotation=60)
        plt.yticks(range(len(corrmat_labls)), corrmat_labls)
        fig.colorbar(cax, orientation='vertical')
        plt.subplots_adjust(left=0.025, bottom=0., right=1., top=0.975, wspace=0., hspace=0.)
        plt.show()
