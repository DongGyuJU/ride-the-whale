"""
IC (Information Coefficient) 평가.

- Spearman IC: 예측값과 실제 forward return의 rank 상관계수
- IC IR: IC / std(IC) — 신호 안정성
- 날짜별 IC 시계열로 계산
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def compute_daily_ic(
    pred: pd.Series,
    y: pd.Series,
) -> pd.Series:
    """
    날짜별 Spearman IC.

    Args:
        pred: MultiIndex (date, ticker) 예측값
        y   : MultiIndex (date, ticker) 실제 수익률

    Returns:
        Series[date → IC]
    """
    common = pred.index.intersection(y.index)
    pred = pred.loc[common]
    y = y.loc[common]

    daily_ic = {}
    for date, grp_pred in pred.groupby(level="date"):
        grp_y = y.loc[date] if date in y.index.get_level_values("date") else None
        if grp_y is None or len(grp_pred) < 5:
            continue
        common_t = grp_pred.index.get_level_values("ticker").intersection(
            grp_y.index.get_level_values("ticker") if isinstance(grp_y.index, pd.MultiIndex)
            else grp_y.index
        )
        if len(common_t) < 5:
            continue
        try:
            p = grp_pred.droplevel("date") if "date" in grp_pred.index.names else grp_pred
            g = grp_y.droplevel("date") if isinstance(grp_y.index, pd.MultiIndex) else grp_y
            p, g = p.align(g, join="inner")
            mask = ~(p.isna() | g.isna())
            if mask.sum() < 5:
                continue
            ic = spearmanr(p[mask], g[mask]).statistic
            daily_ic[date] = ic
        except Exception:
            continue

    return pd.Series(daily_ic, name="IC")


def summarize_ic(ic_series: pd.Series, label: str = "") -> dict:
    """IC 요약 통계."""
    clean = ic_series.dropna()
    if len(clean) == 0:
        return {}

    mean_ic = clean.mean()
    std_ic = clean.std()
    ir = mean_ic / (std_ic + 1e-10)
    t_stat = mean_ic / (std_ic / np.sqrt(len(clean)) + 1e-10)
    pct_pos = (clean > 0).mean()

    result = {
        "IC_mean": mean_ic,
        "IC_std": std_ic,
        "IC_IR": ir,
        "IC_t": t_stat,
        "IC_pct_positive": pct_pos,
        "n_days": len(clean),
    }

    tag = f"[{label}] " if label else ""
    logger.info(f"\n{tag}IC 요약:")
    logger.info(f"  IC Mean     : {mean_ic:.4f}")
    logger.info(f"  IC Std      : {std_ic:.4f}")
    logger.info(f"  IC IR       : {ir:.4f}")
    logger.info(f"  t-stat      : {t_stat:.2f}")
    logger.info(f"  IC>0 비율   : {pct_pos:.1%}")
    logger.info(f"  거래일 수   : {len(clean)}")

    return result
