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

epoch = datetime.utcfromtimestamp(0)

train_csvpath = os.path.join(data_dir, 'train_merged.csv')
train_data = pd.read_csv(train_csvpath,
    dtype={
        'StateHoliday': str,
    },
    parse_dates=['Date'],
    date_parser=strtodate)

num_stores = 1115
indices = np.arange(1, num_stores+1)
np.random.shuffle(indices)

split_size = 115
selector = np.zeros(len(train_data), dtype=bool)
for i in range(split_size):
    selector = selector | (train_data['Store'] == indices[i])

train_data.loc[selector].to_csv(os.path.join(data_dir, '115_split.csv'), date_format='%d/%m/%Y', index=False)
train_data.loc[~selector].to_csv(os.path.join(data_dir, '1000_split.csv'), date_format='%d/%m/%Y', index=False)
