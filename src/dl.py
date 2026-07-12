from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .features import HOLIDAY_DATES

HORIZON = 39          # length of the Kaggle test period, in weeks
INPUT_LEN = 52        # one full year of history as input


# --------------------------------------------------------------- data prep
def build_series_matrix(train_raw: pd.DataFrame):
    """
    :param train_raw: raw train data
    :return:
    values: missing weeks <br>
    mask: if week was observed then 1 <br>
    keys: DataFrame(Store, Dept) aligned with the rows <br>
    dates: DatetimeIndex of the T columns
    """
    df = train_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="W-FRI")
    wide = df.pivot_table(index=["Store", "Dept"], columns="Date",
                          values="Weekly_Sales", aggfunc="sum")
    wide = wide.reindex(columns=dates)
    mask = wide.notna().to_numpy(dtype=np.float32)
    values = wide.fillna(0.0).to_numpy(dtype=np.float32)
    keys = wide.index.to_frame(index=False)
    return values, mask, keys, dates


def holiday_weight_vector(dates: pd.DatetimeIndex) -> np.ndarray:
    all_hol = pd.DatetimeIndex(
        [d for v in HOLIDAY_DATES.values() for d in v])
    return np.where(dates.isin(all_hol), 5.0, 1.0).astype(np.float32)


def mean_abs_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = np.abs(values).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    return np.maximum(s, 1.0).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, scaled: np.ndarray, mask: np.ndarray,
                 hol_w: np.ndarray, input_len: int = INPUT_LEN,
                 horizon: int = HORIZON):
        self.scaled, self.mask = scaled, mask
        self.hol_w = hol_w
        self.L, self.H = input_len, horizon
        T = scaled.shape[1]
        ends = range(self.L, T - self.H + 1)
        self.index = [(i, e) for e in ends
                      for i in range(scaled.shape[0])
                      if mask[i, e:e + self.H].sum() > 0]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, e = self.index[k]
        x = self.scaled[i, e - self.L:e]
        y = self.scaled[i, e:e + self.H]
        w = self.mask[i, e:e + self.H] * self.hol_w[e:e + self.H]
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(w.astype(np.float32)))


def weighted_mae(pred, target, weight):
    return (weight * (pred - target).abs()).sum() / weight.sum().clamp(min=1.0)



class DLinear(nn.Module):

    def __init__(self, input_len: int = INPUT_LEN, horizon: int = HORIZON,
                 kernel: int = 13):
        super().__init__()
        self.kernel = kernel
        self.linear_trend = nn.Linear(input_len, horizon)
        self.linear_season = nn.Linear(input_len, horizon)

    def _trend(self, x):
        pad_l = (self.kernel - 1) // 2
        pad_r = self.kernel - 1 - pad_l
        xp = torch.cat([x[:, :1].expand(-1, pad_l),
                        x,
                        x[:, -1:].expand(-1, pad_r)], dim=1)
        return xp.unfold(1, self.kernel, 1).mean(dim=2)

    def forward(self, x):
        trend = self._trend(x)
        season = x - trend
        return self.linear_trend(trend) + self.linear_season(season)


class NBeatsBlock(nn.Module):
    def __init__(self, input_len, horizon, width=256, n_layers=4, theta_dim=64):
        super().__init__()
        layers, d = [], input_len
        for _ in range(n_layers):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(width, theta_dim * 2)
        self.backcast_basis = nn.Linear(theta_dim, input_len, bias=False)
        self.forecast_basis = nn.Linear(theta_dim, horizon, bias=False)
        self.theta_dim = theta_dim

    def forward(self, x):
        theta = self.theta(self.fc(x))
        return (self.backcast_basis(theta[:, :self.theta_dim]),
                self.forecast_basis(theta[:, self.theta_dim:]))


class NBeats(nn.Module):
    def __init__(self, input_len: int = INPUT_LEN, horizon: int = HORIZON,
                 n_blocks: int = 6, width: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(input_len, horizon, width) for _ in range(n_blocks)])

    def forward(self, x):
        residual, forecast = x, 0.0
        for block in self.blocks:
            backcast, block_fc = block(residual)
            residual = residual - backcast
            forecast = forecast + block_fc
        return forecast



def train_model(model, train_ds, epochs=30, batch_size=1024,
                lr=1e-3, device=None, log_fn=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y, w in train_dl:
            x, y, w = x.to(device), y.to(device), w.to(device)
            loss = weighted_mae(model(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        train_loss = tot / n
        if log_fn:
            log_fn(epoch, train_loss, None)
    return model


try:  # soft import: keeps src.dl usable in environments without MLflow
    from mlflow.pyfunc import PythonModel as _PyfuncBase
except ImportError:
    _PyfuncBase = object


class GlobalForecastPyfunc(_PyfuncBase):
    def __init__(self, model, history_scaled, scales, keys, last_train_date,
                 horizon: int = HORIZON):
        self.model = model.cpu().eval()
        self.history_scaled = history_scaled.astype(np.float32)
        self.scales = scales
        self.horizon = horizon
        self.last_train_date = pd.Timestamp(last_train_date)
        self.key_to_row = {(int(s), int(d)): i
                           for i, (s, d) in enumerate(zip(keys["Store"],
                                                          keys["Dept"]))}

    def _forecast_matrix(self) -> np.ndarray:
        with torch.no_grad():
            pred = self.model(torch.from_numpy(self.history_scaled)).numpy()
        return pred * self.scales[:, None]

    def predict(self, context, model_input: pd.DataFrame, params=None):
        fc = self._forecast_matrix()
        dates = pd.to_datetime(model_input["Date"])
        step = ((dates - self.last_train_date).dt.days // 7 - 1).to_numpy()
        rows = np.array([self.key_to_row.get((int(s), int(d)), -1)
                         for s, d in zip(model_input["Store"],
                                         model_input["Dept"])])
        ok = (rows >= 0) & (step >= 0) & (step < self.horizon)
        out = np.zeros(len(model_input))
        out[ok] = fc[rows[ok], step[ok]]
        return out
