from __future__ import annotations

import numpy as np


def holiday_weights(is_holiday) -> np.ndarray:
    """5 for holiday weeks, 1 otherwise"""
    return np.where(np.asarray(is_holiday).astype(bool), 5.0, 1.0)


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = holiday_weights(is_holiday)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))
