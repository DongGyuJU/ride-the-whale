"""
피처 미분값 (변화율) 추가
============================

이론적 근거:
  - Ehsani & Linnainmaa (2022, JF): 팩터 모멘텀
    → 피처의 변화율이 수준값보다 예측력 높음
  - FM 결과 확인:
    sup_fcum20 (수준값) t=-7.09 음수
    sup_fmom_chg (변화율) t=+3.26 양수
    → 같은 정보인데 미분이 반대 방향

생성 피처:
  d5_{col}   — 5일 변화 (단기 가속도)
  d20_{col}  — 20일 변화 (중기 추세 변화)

대상:
  FM 유의 피처 + 매크로 피처
  수준값과 미분값 동시에 넣어서 EN이 선택
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 미분 적용할 피처 (FM 유의 + 매크로)
DERIVATIVE_TARGETS = [
    # 기술적 (FM 유의)
    "tech_52w_pos", "tech_bb_pos", "tech_tv_ratio",
    "tech_mom20", "tech_vol20", "tech_mom60",
    # 수급 (FM 유의)
    "sup_fmom_chg", "sup_fnet_rank", "sup_fcum20", "sup_fstreak",
    "sup_fcum10", "sup_icum20",
    # 매크로
    "macro_rate10_chg20", "macro_yield_spread",
    "macro_kospi_mom60",
]

DERIVATIVE_WINDOWS = [5, 20]


def add_derivative_features(
    feat_df: pd.DataFrame,
    feat_cols: list[str],
    targets: list[str] | None = None,
    windows: list[int] = DERIVATIVE_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    피처의 rolling 변화 (1차 미분) 추가.

    Δ_w X_t = X_t - X_{t-w}  (w일 변화)

    Args:
        feat_df  : 피처 DataFrame (MultiIndex date/ticker)
        feat_cols: 기존 피처 컬럼 리스트
        targets  : 미분 적용할 피처 (None=DERIVATIVE_TARGETS)
        windows  : 미분 윈도우

    Returns:
        (업데이트된 feat_df, 전체 feat_cols)
    """
    out = feat_df.copy()
    new_cols = []
    targets = targets or DERIVATIVE_TARGETS

    # 실제 존재하는 컬럼만
    valid_targets = [t for t in targets if t in out.columns]
    missing = set(targets) - set(valid_targets)
    if missing:
        logger.debug(f"  미분 대상 누락 {len(missing)}개: {missing}")

    for col in valid_targets:
        for w in windows:
            new_col = f"d{w}_{col}"
            # ticker 별 w일 차분
            out[new_col] = out.groupby(level="ticker")[col].transform(
                lambda s, w=w: s.diff(w)
            )
            new_cols.append(new_col)

    logger.info(f"  미분 피처: {len(new_cols)}개 추가 "
                f"({len(valid_targets)}개 피처 × {len(windows)}개 윈도우)")
    return out, feat_cols + new_cols
