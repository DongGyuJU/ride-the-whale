"""
G. 어닝 이벤트 피처 (변화율 기반)
=====================================

Ehsani & Linnainmaa (2022) 팩터 모멘텀:
  수익률이 아닌 이익의 변화율/가속도가 더 강한 신호

피처:
  earn_yoy       — YOY 영업이익 변화율
  earn_accel     — 이익 성장의 가속도 (2차 미분) ← 핵심
  earn_qoq       — QOQ 변화율 (단기)
  earn_rev_yoy   — 매출 YOY 변화율

미래정보 방지:
  report_date = 분기말 + 45일 (공시 지연 반영)
  → 패널에 적용 시 report_date 이후부터만 사용
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

EARNINGS_FEATURES = [
    "earn_yoy",       # t=-3.35 ✅
    "earn_accel",     # t=+6.46 ✅ 핵심
    "earn_rev_yoy",   # t=-7.75 ✅
    # earn_qoq      t=-0.54 ❌ 제거
]


def load_earnings_data(earn_dir: str | Path) -> pd.DataFrame | None:
    earn_path = Path(earn_dir) / "kospi_earnings.parquet"
    if not earn_path.exists():
        logger.warning(f"  어닝 데이터 없음: {earn_path}")
        return None
    df = pd.read_parquet(earn_path)
    df["ticker"]      = df["ticker"].astype(str).str.zfill(6)
    df["report_date"] = pd.to_datetime(df["report_date"])
    logger.info(f"  어닝 데이터: {df.shape} | "
                f"{df['fiscal_quarter'].min()} ~ {df['fiscal_quarter'].max()}")
    return df


def add_earnings_features(
    panel: pd.DataFrame,
    earn_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    패널에 어닝 피처 추가.
    각 종목의 가장 최근 공시 데이터를 해당 날짜에 매핑.
    """
    out = panel.copy()
    for col in EARNINGS_FEATURES:
        out[col] = np.nan

    if earn_df is None or earn_df.empty:
        return out

    dates   = out.index.get_level_values("date")
    tickers = out.index.get_level_values("ticker")

    # 컬럼 매핑
    col_map = {
        "earn_yoy":     "op_yoy",
        "earn_accel":   "op_accel",
        "earn_qoq":     "op_qoq",
        "earn_rev_yoy": "rev_yoy",
    }

    for tkr, grp in earn_df.groupby("ticker"):
        grp = grp.sort_values("report_date").reset_index(drop=True)
        ticker_mask = (tickers == tkr)
        if not ticker_mask.any():
            continue

        for idx, row in grp.iterrows():
            report_dt = row["report_date"]

            # 다음 공시일
            if idx + 1 < len(grp):
                next_dt = grp.loc[idx + 1, "report_date"]
            else:
                next_dt = pd.Timestamp("2099-12-31")

            # 공시일 ~ 다음 공시일 사이 날짜에 적용
            date_mask = ticker_mask & (dates >= report_dt) & (dates < next_dt)
            if not date_mask.any():
                continue

            for feat_col, src_col in col_map.items():
                if src_col in row.index and not pd.isna(row[src_col]):
                    out.loc[date_mask, feat_col] = float(row[src_col])

    coverage = out[EARNINGS_FEATURES].notna().mean()
    logger.info(f"  어닝 피처 커버리지: "
                f"{coverage.mean():.1%} 평균")
    return out
