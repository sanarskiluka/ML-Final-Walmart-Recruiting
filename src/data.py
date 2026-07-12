from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

COMPETITION = "walmart-recruiting-store-sales-forecasting"

_SEARCH_ROOTS = [
    Path("."),
    Path("data"),
    Path(f"data/{COMPETITION}"),
    Path(f"/content/{COMPETITION}"),   # collab is gamo
    Path(f"/content/data/{COMPETITION}"),
]


def find_data_dir() -> Path:
    env = os.environ.get("WALMART_DATA_DIR")
    roots = ([Path(env)] if env else []) + _SEARCH_ROOTS
    for root in roots:
        for candidate in (root / "train.csv", root / "train.csv" / "train.csv"):
            if candidate.is_file():
                return root
    raise FileNotFoundError("File not found. set WALMART_DATA_DIR environment variable where the data is.")


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if path.is_dir():
        path = path / name
    with open(path, newline=None) as fh:
        return pd.read_csv(fh)


def load_raw(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """ Loads raw dataframes """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    out = {
        "train": _read(data_dir, "train.csv"),
        "test": _read(data_dir, "test.csv"),
        "features": _read(data_dir, "features.csv"),
        "stores": _read(data_dir, "stores.csv"),
    }
    for key in ("train", "test", "features"):
        out[key]["Date"] = pd.to_datetime(out[key]["Date"])
    return out


def submission_ids(test: pd.DataFrame) -> pd.Series:
    return (
        test["Store"].astype(str)
        + "_" + test["Dept"].astype(str)
        + "_" + test["Date"].dt.strftime("%Y-%m-%d")
    )
