"""
시간 기반 데이터 분할.

2018-2021: train (4년)
2022:      val   (1년)
2023-2024: test  (2년)
"""
from __future__ import annotations
import pandas as pd


SPLIT = {
    "train": ("2018-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2024-12-31"),
}


def temporal_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Args:
        X: MultiIndex (date, ticker) 피처
        y: MultiIndex (date, ticker) 레이블

    Returns:
        {"train": (X_tr, y_tr), "val": ..., "test": ...}
    """
    dates = X.index.get_level_values("date")
    splits = {}

    for name, (start, end) in SPLIT.items():
        mask = (dates >= start) & (dates <= end)
        X_s = X[mask].dropna()
        y_s = y[mask].loc[X_s.index].dropna()
        # 공통 인덱스
        common = X_s.index.intersection(y_s.index)
        splits[name] = (X_s.loc[common], y_s.loc[common])

    for name, (xs, ys) in splits.items():
        d = xs.index.get_level_values("date")
        print(f"  {name}: {d.min().date()} ~ {d.max().date()}  "
              f"({len(xs):,} rows, "
              f"{xs.index.get_level_values('ticker').nunique()} tickers)")

    return splits
