"""
C. 공매도 피처
================

컬럼: short_volume, total_volume, short_ratio_pct, short_ratio

피처:
  short_ratio     — 공매도 비율 (이미 계산됨)
  short_chg5      — 5일 공매도 비율 변화 (상승 = 약세 심화)
  short_bal_rank  — 공매도 비율 시장 내 순위
  short_squeeze   — 스퀴즈 압박 (공매도 많은데 가격 상승)

FM 가설:
  short_ratio↑   → 약세 신호 (음의 IC 예상)
  short_squeeze↑ → 강세 신호 (커버링 매수 유발)
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

SHORT_FEATURES = [
    "short_ratio",   # t=-3.02 ✅
    # short_chg5    t=-0.81 ❌ 제거
    # short_bal_rank t=+1.25 ❌ 제거
    # short_squeeze  t=-1.09 ❌ 제거
]


def load_short_data(short_dir: str | Path) -> pd.DataFrame | None:
    short_path = Path(short_dir) / "kospi_short.parquet"
    if not short_path.exists():
        logger.warning(f"  공매도 데이터 없음: {short_path}")
        return None
    df = pd.read_parquet(short_path)
    logger.info(f"  공매도 데이터: {df.shape}")
    return df


def add_short_features(
    panel: pd.DataFrame,
    short_df: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()
    grp = out.groupby(level="ticker", group_keys=False)

    # short_ratio 병합
    common = out.index.intersection(short_df.index)
    if len(common) == 0:
        for col in SHORT_FEATURES:
            out[col] = np.nan
        logger.warning("  공매도 데이터 인덱스 겹침 없음")
        return out

    out.loc[common, "short_ratio"] = short_df.loc[common, "short_ratio"]

    # 5일 변화
    out["short_chg5"] = grp["short_ratio"].transform(
        lambda s: s.diff(5)
    )

    # 시장 내 순위
    out["short_bal_rank"] = out.groupby(level="date")["short_ratio"].transform(
        lambda s: s.rank(pct=True, na_option="keep")
    )

    # 스퀴즈 압박: 공매도 많은데 최근 가격 상승
    if "close" in out.columns:
        mom5 = grp["close"].transform(lambda s: s.pct_change(5))
        out["short_squeeze"] = out["short_bal_rank"] * mom5
    else:
        out["short_squeeze"] = np.nan

    coverage = out["short_ratio"].notna().mean()
    logger.info(f"  공매도 피처 커버리지: {coverage:.1%}")
    return out
