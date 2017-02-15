import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

def load_train_data(data_dir='data/'):
    csv_path = os.path.join(data_dir, 'train_merged.csv')
    data = pd.read_csv(csv_path,
        dtype={
            'StateHoliday': str,
        },
        parse_dates=['Date'],
        date_parser=strtodate)

    return data

def load_splitted_data(data_dir='data/'):
    csv_path = os.path.join(data_dir, '1000_split.csv')
    split_data = pd.read_csv(csv_path,
        dtype={
            'StateHoliday': str,
        },
        parse_dates=['Date'],
        date_parser=strtodate)

    csv_path2 = os.path.join(data_dir, '115_split.csv')
    split_data2 = pd.read_csv(csv_path2,
        dtype={
            'StateHoliday': str,
        },
        parse_dates=['Date'],
        date_parser=strtodate)

    return split_data, split_data2

def load_test_data(data_dir='data/'):
    csv_path = os.path.join(data_dir, 'test_merged.csv')
    data = pd.read_csv(csv_path,
        dtype={
            'StateHoliday': str,
        },
        parse_dates=['Date'],
        date_parser=strtodate)

    return data
