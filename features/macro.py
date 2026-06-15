"""
거시경제 피처 (macro_*)
========================

데이터:
  - KODEX 200 ETF (069500) → KOSPI 레짐
  - 국고채 10년 ETF (148070) → 장기 금리
  - 국고채 3년 ETF (114820) → 단기 금리

피처:
  macro_kospi_regime   — KOSPI 200일선 위(1) / 아래(0)
  macro_kospi_mom60    — KOSPI 60일 모멘텀 (시장 방향)
  macro_rate10_chg20   — 10년물 20일 변화 (상승=금리 상승 레짐)
  macro_yield_spread   — 10년-3년 스프레드 (경기 선행)
  macro_spread_chg20   — 스프레드 20일 변화
  macro_rate_regime    — 금리 하락(1) / 상승(0) 레짐

사용법:
  macro_df = load_macro(macro_dir)
  panel = add_macro_features(panel, macro_df)
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

MACRO_FEATURES = [
    "macro_kospi_regime",
    "macro_kospi_mom60",
    "macro_rate10_chg20",
    "macro_yield_spread",
    "macro_spread_chg20",
    "macro_rate_regime",
]


def load_macro(macro_dir: str | Path) -> pd.DataFrame:
    """
    매크로 parquet 파일 로드 → 날짜별 피처 DataFrame.

    Returns:
        DataFrame[macro_*] indexed by date
    """
    macro_dir = Path(macro_dir)
    log = logging.getLogger(__name__)

    # ETF 종가 로드
    bond10 = pd.read_parquet(macro_dir / "bond_10y.parquet")["종가"].rename("bond10")
    bond3  = pd.read_parquet(macro_dir / "bond_3y.parquet")["종가"].rename("bond3")

    # KOSPI (KODEX 200 ETF 또는 지수)
    kospi_path = macro_dir / "kospi_index.parquet"
    kospi = pd.read_parquet(kospi_path)["종가"].rename("kospi")

    # 날짜 인덱스 통일
    df = pd.DataFrame({"bond10": bond10, "bond3": bond3, "kospi": kospi})
    df = df.sort_index().ffill()  # 결측일 전일값으로 채움

    # ── 피처 계산 ──────────────────────────────────

    # 1. KOSPI 레짐 (200일선 기준)
    ma200 = df["kospi"].rolling(200, min_periods=100).mean()
    df["macro_kospi_regime"] = (df["kospi"] > ma200).astype(float)

    # 2. KOSPI 60일 모멘텀
    df["macro_kospi_mom60"] = df["kospi"].pct_change(60)

    # 3. 10년물 금리 변화 (ETF 역방향: ETF 하락 = 금리 상승)
    # ETF 수익률의 부호 반전 = 금리 변화 방향
    df["macro_rate10_chg20"] = -df["bond10"].pct_change(20)

    # 4. 장단기 스프레드 (10년-3년)
    # ETF 가격 역수 비율로 스프레드 근사
    # bond10 하락 = 장기금리 상승, bond3 하락 = 단기금리 상승
    df["macro_yield_spread"] = df["bond3"] / df["bond10"] - 1
    # 양수 = 단기>장기 (역전, 경기침체 신호)
    # 음수 = 정상 (장기>단기)

    # 5. 스프레드 변화
    df["macro_spread_chg20"] = df["macro_yield_spread"].diff(20)

    # 6. 금리 레짐 (하락=1 / 상승=0)
    # 10년물 ETF 20일 모멘텀 양수 = ETF 상승 = 금리 하락
    df["macro_rate_regime"] = (df["bond10"].pct_change(20) > 0).astype(float)

    # 피처만 반환
    result = df[MACRO_FEATURES].copy()
    log.info(f"[macro] 로드 완료: {len(result)}일 | "
             f"{result.index.min().date()} ~ {result.index.max().date()}")
    log.info(f"[macro] 금리 하락 레짐 비율: {result['macro_rate_regime'].mean():.1%}")
    log.info(f"[macro] KOSPI 강세 레짐 비율: {result['macro_kospi_regime'].mean():.1%}")
    return result


def add_macro_features(
    panel: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    패널에 매크로 피처 추가.
    날짜 기준으로 left join (종목별로 같은 매크로 값 적용).

    Args:
        panel   : MultiIndex (date, ticker) 패널
        macro_df: load_macro() 반환값

    Returns:
        매크로 피처가 추가된 패널
    """
    out = panel.copy()
    dates = out.index.get_level_values("date")

    # 매크로 데이터를 패널 날짜에 맞게 reindex (ffill)
    macro_reindexed = macro_df.reindex(
        dates.unique().sort_values()
    ).ffill()

    # 각 날짜-종목 행에 매크로 값 붙이기
    for col in MACRO_FEATURES:
        date_map = macro_reindexed[col].to_dict()
        out[col] = dates.map(date_map)

    return out
