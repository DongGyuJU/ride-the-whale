"""
Long-Short 포트폴리오 평가.

- 상위 10% 매수 (long), 하위 10% 매도 (short)
- 동일가중 (equal-weight)
- 거래비용 미반영 (gross return)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_long_short_returns(
    pred: pd.Series,
    fwd_ret: pd.Series,
    q: float = 0.1,
) -> pd.DataFrame:
    """
    Args:
        pred   : MultiIndex (date, ticker) 예측 점수
        fwd_ret: MultiIndex (date, ticker) 실제 수익률
        q      : 상/하위 분위수 (0.1 = 10%)

    Returns:
        DataFrame[long_ret, short_ret, ls_ret] indexed by date
    """
    common = pred.index.intersection(fwd_ret.index)
    pred = pred.loc[common]
    fwd_ret = fwd_ret.loc[common]

    daily_results = []

    for date, grp_pred in pred.groupby(level="date"):
        try:
            grp_ret = fwd_ret.loc[date]
        except KeyError:
            continue

        p = grp_pred.droplevel("date") if "date" in grp_pred.index.names else grp_pred
        r = grp_ret.droplevel("date") if isinstance(grp_ret.index, pd.MultiIndex) else grp_ret
        p, r = p.align(r, join="inner")
        mask = ~(p.isna() | r.isna())
        p, r = p[mask], r[mask]

        if len(p) < 10:
            continue

        lo_q = p.quantile(q)
        hi_q = p.quantile(1 - q)

        long_ret  = r[p >= hi_q].mean()
        short_ret = r[p <= lo_q].mean()
        ls_ret    = long_ret - short_ret

        daily_results.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": (p >= hi_q).sum(),
            "n_short": (p <= lo_q).sum(),
        })

    if not daily_results:
        return pd.DataFrame()

    df = pd.DataFrame(daily_results).set_index("date").sort_index()
    return df


def summarize_long_short(ls_df: pd.DataFrame, label: str = "") -> dict:
    """Long-Short 포트폴리오 요약."""
    if ls_df.empty:
        return {}

    ls = ls_df["ls_ret"].dropna()
    ann_ret = ls.mean() * 252
    ann_vol = ls.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-10)
    cum = (1 + ls).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()

    result = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "hit_rate": (ls > 0).mean(),
    }

    tag = f"[{label}] " if label else ""
    logger.info(f"\n{tag}Long-Short 요약 (gross, 거래비용 미반영):")
    logger.info(f"  연환산 수익률  : {ann_ret:.2%}")
    logger.info(f"  연환산 변동성  : {ann_vol:.2%}")
    logger.info(f"  Sharpe         : {sharpe:.2f}")
    logger.info(f"  Max Drawdown   : {drawdown:.2%}")
    logger.info(f"  Hit Rate       : {(ls > 0).mean():.1%}")

    return result
