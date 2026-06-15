"""
피처 파이프라인 v2 — FM 기반 재설계 피처 사용
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from features.technical_v2 import add_technical_features_v2, TECHNICAL_FEATURES_V2
from features.supply_v2 import add_supply_features_v2, SUPPLY_FEATURES_V2

logger = logging.getLogger(__name__)


def cross_sectional_zscore(df, feature_cols, clip=3.0):
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(level="date")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-10)
        ).clip(-clip, clip)
    return out


def build_features_v2(
    panel: pd.DataFrame,
    mode: str = "full",
    universe: list[str] | None = None,
    zscore: bool = True,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:

    if cache_path and Path(cache_path).exists():
        logger.info(f"[pipeline_v2] 캐시 로드: {cache_path}")
        feat_df = pd.read_parquet(cache_path)
        feat_cols = [c for c in feat_df.columns
                     if c.startswith("tech_") or c.startswith("sup_")]
        return feat_df, feat_cols

    logger.info(f"[pipeline_v2] 피처 빌드 시작 (mode={mode})")

    if universe:
        tickers = panel.index.get_level_values("ticker")
        panel = panel[tickers.isin(universe)]

    # 기술적 피처 v2
    logger.info("[pipeline_v2] 기술적 피처 v2 계산 중...")
    df = add_technical_features_v2(panel)
    feat_cols = [c for c in TECHNICAL_FEATURES_V2 if c in df.columns]

    # 수급 피처 v2
    if mode == "full":
        supply_cols = [c for c in ["foreign_net", "inst_net"] if c in df.columns]
        if supply_cols:
            logger.info("[pipeline_v2] 수급 피처 v2 계산 중...")
            df = add_supply_features_v2(df)
            sup_cols = [c for c in SUPPLY_FEATURES_V2 if c in df.columns]
            feat_cols = feat_cols + sup_cols
            logger.info(f"[pipeline_v2] 수급 피처: {len(sup_cols)}개")

    logger.info(f"[pipeline_v2] 총 피처: {len(feat_cols)}개")

    if zscore:
        df = cross_sectional_zscore(df, feat_cols)

    keep = feat_cols + [c for c in ["close", "trade_value"] if c in df.columns]
    feat_df = df[[c for c in keep if c in df.columns]]

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(cache_path, compression="snappy")
        logger.info(f"[pipeline_v2] 캐시 저장: {cache_path}")

    logger.info(f"[pipeline_v2] 완료: {feat_df.shape}")
    return feat_df, feat_cols
