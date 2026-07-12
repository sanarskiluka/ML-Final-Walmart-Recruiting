from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2012-10-26")     # last date in train.csv
DEV_TRAIN_END = pd.Timestamp("2011-10-28")  # one year before TRAIN_END
DEV_VALID_START = pd.Timestamp("2011-11-04")
DEV_VALID_END = pd.Timestamp("2012-07-27")  # 39 weeks


def year_back_split(dates: pd.Series):
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    train_idx = np.where(dates <= DEV_TRAIN_END)[0]
    valid_idx = np.where((dates >= DEV_VALID_START) & (dates <= DEV_VALID_END))[0]
    return train_idx, valid_idx


def time_folds(dates: pd.Series, n_folds: int = 3, valid_weeks: int = 13):
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    for k in range(n_folds, 0, -1):
        valid_end = TRAIN_END - pd.Timedelta(weeks=(k - 1) * valid_weeks)
        valid_start = valid_end - pd.Timedelta(weeks=valid_weeks - 1)
        train_idx = np.where(dates < valid_start)[0]
        valid_idx = np.where((dates >= valid_start) & (dates <= valid_end))[0]
        yield train_idx, valid_idx
