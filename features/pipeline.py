"""
피처 파이프라인.

build_features(panel, mode) → feature_df

mode:
    'ohlcv'  : 기술적 피처만 (baseline)
    'full'   : 기술적 + 수급 피처 (main)
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from features.technical import add_technical_features, TECHNICAL_FEATURES
from features.supply import add_supply_features, SUPPLY_FEATURES

logger = logging.getLogger(__name__)


def cross_sectional_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    clip: float = 3.0,
) -> pd.DataFrame:
    """
    날짜별 cross-sectional z-score 정규화.
    - 각 날짜에 대해 종목 간 분포를 표준화
    - ±clip 으로 winsorize
    """
    out = df.copy()
    date_grp = out.groupby(level="date", group_keys=False)

    for col in feature_cols:
        if col not in out.columns:
            continue
        out[col] = date_grp[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-10)
        ).clip(-clip, clip)

    return out


def build_features(
    panel: pd.DataFrame,
    mode: str = "full",
    universe: list[str] | None = None,
    zscore: bool = True,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Args:
        panel   : MultiIndex (date, ticker) 패널
        mode    : 'ohlcv' | 'full'
        universe: 종목 필터 리스트 (None = 전체)
        zscore  : cross-sectional z-score 적용 여부
        cache_path: 캐시 경로 (있으면 저장/로드)

    Returns:
        (feature_df, feature_cols)
        feature_df: 피처 컬럼만 있는 MultiIndex DataFrame
    """
    # 캐시 확인
    if cache_path and Path(cache_path).exists():
        logger.info(f"[pipeline] 캐시 로드: {cache_path}")
        feat_df = pd.read_parquet(cache_path)
        feat_cols = [c for c in feat_df.columns
                     if c.startswith("tech_") or c.startswith("sup_")]
        return feat_df, feat_cols

    logger.info(f"[pipeline] 피처 빌드 시작 (mode={mode})")

    # 유니버스 필터
    if universe is not None:
        tickers = panel.index.get_level_values("ticker")
        panel = panel[tickers.isin(universe)]
        logger.info(f"[pipeline] 유니버스 필터: {len(universe)} 종목")

    # 기술적 피처
    logger.info("[pipeline] 기술적 피처 계산 중...")
    df = add_technical_features(panel)
    feat_cols = [c for c in TECHNICAL_FEATURES if c in df.columns]

    # 수급 피처
    if mode == "full":
        supply_cols_present = [c for c in ["foreign_net", "inst_net"] if c in df.columns]
        if supply_cols_present:
            logger.info("[pipeline] 수급 피처 계산 중...")
            df = add_supply_features(df)
            sup_cols = [c for c in SUPPLY_FEATURES if c in df.columns]
            feat_cols = feat_cols + sup_cols
            logger.info(f"[pipeline] 수급 피처: {len(sup_cols)}개")
        else:
            logger.warning("[pipeline] 수급 컬럼 없음 — OHLCV 모드로 fallback")

    logger.info(f"[pipeline] 총 피처: {len(feat_cols)}개")

    # Cross-sectional z-score
    if zscore:
        logger.info("[pipeline] Cross-sectional z-score 정규화 중...")
        df = cross_sectional_zscore(df, feat_cols)

    # 피처 컬럼만 추출 (원본 OHLCV + 수급 원데이터 제거)
    keep_cols = feat_cols + [c for c in ["close", "trade_value", "market_cap"]
                              if c in df.columns]
    feat_df = df[[c for c in keep_cols if c in df.columns]]

    # 캐시 저장
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(cache_path, compression="snappy")
        logger.info(f"[pipeline] 캐시 저장: {cache_path}")

    logger.info(f"[pipeline] 완료: {feat_df.shape}")
    return feat_df, feat_cols
