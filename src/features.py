from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HOLIDAY_DATES = {
    "SuperBowl":    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "LaborDay":     ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas":    ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}
HOLIDAY_DATES = {k: pd.to_datetime(v) for k, v in HOLIDAY_DATES.items()}

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]


def _weeks_to_nearest(
        dates: pd.Series,
        holiday_dates: pd.DatetimeIndex,
        clip: int
) -> np.ndarray:
    diffs = (dates.values[:, None] - holiday_dates.values[None, :])
    diffs_w = diffs / np.timedelta64(7, "D")
    idx = np.abs(diffs_w).argmin(axis=1)
    nearest = diffs_w[np.arange(len(dates)), idx]
    return np.clip(np.round(nearest), -clip, clip).astype(int)


class WalmartFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, features_df: pd.DataFrame | None = None,
                 stores_df: pd.DataFrame | None = None,
                 lags: tuple = (52,),
                 use_markdowns: bool = True):
        self.features_df = features_df
        self.stores_df = stores_df
        self.lags = lags
        self.use_markdowns = use_markdowns

    def fit(self, X: pd.DataFrame, y=None):
        if self.features_df is None or self.stores_df is None:
            raise ValueError("features_df and stores_df should be passed in arguments.")

        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        if y is None:
            if "Weekly_Sales" not in X.columns:
                raise ValueError("y should not be None or dataframe should have 'Weekly_Sales'")
            y = X["Weekly_Sales"]
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)

        side = self.features_df.copy()
        side["Date"] = pd.to_datetime(side["Date"])
        side = side.sort_values(["Store", "Date"])
        for col in ("CPI", "Unemployment"):
            side[col] = side.groupby("Store")[col].transform(
                lambda s: s.ffill().bfill())
        if self.use_markdowns:
            side["has_markdown"] = side[MARKDOWN_COLS].notna().any(axis=1)
            side[MARKDOWN_COLS] = side[MARKDOWN_COLS].fillna(0.0)
            side["md_total"] = side[MARKDOWN_COLS].sum(axis=1)
        self.side_ = side.drop(columns=["IsHoliday"])

        stores = self.stores_df.copy()
        stores["Type"] = stores["Type"].map({"A": 0, "B": 1, "C": 2})
        self.stores_ = stores

        hist = X[["Store", "Dept", "Date"]].copy()
        hist["Weekly_Sales"] = y.values
        self.history_ = hist

        g = hist.groupby(["Store", "Dept"])["Weekly_Sales"]
        self.series_stats_ = g.agg(series_mean="mean", series_std="std",
                                   series_median="median").reset_index()
        self.series_stats_["series_std"] = self.series_stats_["series_std"].fillna(0.0)
        hol = X.get("IsHoliday")
        if hol is not None:
            hol_hist = hist[np.asarray(hol).astype(bool)]
            hm = (hol_hist.groupby(["Store", "Dept"])["Weekly_Sales"]
                  .mean().rename("series_holiday_mean").reset_index())
            self.series_stats_ = self.series_stats_.merge(
                hm, on=["Store", "Dept"], how="left")
        else:
            self.series_stats_["series_holiday_mean"] = np.nan

        self.dept_mean_ = hist.groupby("Dept")["Weekly_Sales"].mean()
        self.store_mean_ = hist.groupby("Store")["Weekly_Sales"].mean()
        self.global_mean_ = float(y.mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["IsHoliday"] = df["IsHoliday"].astype(bool).astype(int)
        df = df.merge(self.stores_, on="Store", how="left")
        df = df.merge(self.side_, on=["Store", "Date"], how="left")

        iso = df["Date"].dt.isocalendar()
        df["week"] = iso.week.astype(int)
        df["year"] = df["Date"].dt.year
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
        df["t"] = (df["Date"] - pd.Timestamp("2010-02-05")).dt.days // 7

        # holidays are weighed 5x more.
        for name, dates in HOLIDAY_DATES.items():
            df[f"is_{name}"] = df["Date"].isin(dates).astype(int)
        df["wk_to_thanksgiving"] = _weeks_to_nearest(
            df["Date"], HOLIDAY_DATES["Thanksgiving"], clip=6)
        df["wk_to_christmas"] = _weeks_to_nearest(
            df["Date"], HOLIDAY_DATES["Christmas"], clip=6)

        for lag in self.lags:
            look = self.history_.copy()
            look["Date"] = look["Date"] + pd.Timedelta(weeks=lag)
            look = look.rename(columns={"Weekly_Sales": f"lag_{lag}"})
            df = df.merge(look, on=["Store", "Dept", "Date"], how="left")

        df = df.merge(self.series_stats_, on=["Store", "Dept"], how="left")
        df["is_new_series"] = df["series_mean"].isna().astype(int)
        fallback = df["Dept"].map(self.dept_mean_)
        fallback = fallback.fillna(df["Store"].map(self.store_mean_))
        fallback = fallback.fillna(self.global_mean_)
        for col in ("series_mean", "series_median", "series_holiday_mean"):
            df[col] = df[col].fillna(fallback)
        df["series_std"] = df["series_std"].fillna(0.0)
        for lag in self.lags:
            col = f"lag_{lag}"
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df["series_mean"])

        feature_cols = [
            "Store", "Dept", "Type", "Size", "IsHoliday",
            "week", "year", "week_sin", "week_cos", "t",
            "Temperature", "Fuel_Price", "CPI", "Unemployment",
            "is_SuperBowl", "is_LaborDay", "is_Thanksgiving", "is_Christmas",
            "wk_to_thanksgiving", "wk_to_christmas",
            "series_mean", "series_std", "series_median", "series_holiday_mean",
            "is_new_series",
        ]
        if self.use_markdowns:
            feature_cols += MARKDOWN_COLS + ["md_total", "has_markdown"]
            df["has_markdown"] = df["has_markdown"].fillna(False).astype(int)
        for lag in self.lags:
            feature_cols += [f"lag_{lag}", f"lag_{lag}_missing"]

        out = df[feature_cols].astype(float)
        self.feature_names_ = feature_cols
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
