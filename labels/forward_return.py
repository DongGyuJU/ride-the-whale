"""
Forward return 레이블.

label_forward_return(panel, horizon) → Series

- horizon 일 후 종가 수익률 (look-ahead: 정상, 이건 y)
- 거래 가능한 날 기준: T+1 매수, T+1+horizon 매도
  (T일 종가 신호 → T+1 시가 매수 가정)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def make_forward_returns(
    panel: pd.DataFrame,
    horizons: list[int] = [5, 10, 20],
    skip: int = 1,
) -> pd.DataFrame:
    """
    Args:
        panel   : MultiIndex (date, ticker), 필수 컬럼: close
        horizons: forward return 기간 (거래일 기준)
        skip    : T+skip 이후 수익률 (1 = T+1 매수, 거래비용 제외)

    Returns:
        DataFrame[fwd5, fwd10, fwd20] — MultiIndex (date, ticker)
    """
    grp = panel.groupby(level="ticker", group_keys=False)
    result = pd.DataFrame(index=panel.index)

    for h in horizons:
        col = f"fwd{h}"
        # T+skip ~ T+skip+h 수익률
        # shift(-skip-h) / shift(-skip) - 1
        result[col] = grp["close"].transform(
            lambda s, h=h, skip=skip:
                s.shift(-skip - h) / s.shift(-skip) - 1
        )
        valid = result[col].notna().sum()
        logger.info(f"[label] fwd{h}: {valid:,} valid rows "
                    f"({result[col].notna().mean():.1%})")

    return result


def winsorize_returns(
    returns: pd.DataFrame,
    q: float = 0.01,
) -> pd.DataFrame:
    """상하위 q% winsorize — 이상치 완화."""
    out = returns.copy()
    for col in out.columns:
        lo = out[col].quantile(q)
        hi = out[col].quantile(1 - q)
        out[col] = out[col].clip(lo, hi)
    return out
