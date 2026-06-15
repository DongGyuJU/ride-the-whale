"""
기술적 피처 (OHLCV 기반, 25개).

모든 피처는 look-ahead bias 없이 rolling 계산.
cross-sectional z-score는 pipeline.py에서 수행.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: MultiIndex (date, ticker) 패널.
            필수 컬럼: close, open, high, low, volume, trade_value

    Returns:
        피처 컬럼이 추가된 DataFrame.
        새 컬럼 prefix: 'tech_'
    """
    out = df.copy()

    # ── 종목별로 계산 (groupby ticker)
    grp = out.groupby(level="ticker", group_keys=False)

    close = out["close"]
    volume = out["volume"]
    high = out["high"]
    low = out["low"]
    trade_value = out["trade_value"]

    # ── 1. 모멘텀 (수익률)
    for w in [5, 10, 20, 60]:
        out[f"tech_mom{w}"] = grp["close"].transform(
            lambda s: s.pct_change(w)
        )

    # ── 2. 단기 반전 (1일)
    out["tech_mom1"] = grp["close"].transform(lambda s: s.pct_change(1))

    # ── 3. 이동평균 대비 괴리율
    for w in [5, 20, 60]:
        ma = grp["close"].transform(lambda s, w=w: s.rolling(w).mean())
        out[f"tech_ma{w}_dev"] = (close - ma) / (ma + 1e-10)

    # ── 4. RSI (14일)
    def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)

    out["tech_rsi14"] = grp["close"].transform(_rsi)

    # ── 5. Bollinger Band 위치 (20일)
    def _bb_pos(s: pd.Series, n: int = 20) -> pd.Series:
        ma = s.rolling(n).mean()
        std = s.rolling(n).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (s - lower) / (upper - lower + 1e-10)

    out["tech_bb_pos"] = grp["close"].transform(_bb_pos)

    # ── 6. 변동성 (20일 수익률 std)
    out["tech_vol20"] = grp["close"].transform(
        lambda s: s.pct_change().rolling(20).std()
    )

    # ── 7. 거래량 모멘텀 (거래량 / 20일 평균)
    out["tech_vol_ratio"] = grp["volume"].transform(
        lambda s: s / (s.rolling(20).mean() + 1e-10)
    )

    # ── 8. 거래대금 모멘텀
    out["tech_tv_ratio"] = grp["trade_value"].transform(
        lambda s: s / (s.rolling(20).mean() + 1e-10)
    )

    # ── 9. 고저 스프레드 (변동성 프록시)
    out["tech_hl_spread"] = (high - low) / (close + 1e-10)

    # ── 10. 종가 위치 (당일 고저 대비)
    out["tech_close_pos"] = (close - low) / (high - low + 1e-10)

    # ── 11. 52주 고저 대비 위치
    def _rank_pos(s: pd.Series, n: int = 252) -> pd.Series:
        roll_min = s.rolling(n, min_periods=60).min()
        roll_max = s.rolling(n, min_periods=60).max()
        return (s - roll_min) / (roll_max - roll_min + 1e-10)

    out["tech_52w_pos"] = grp["close"].transform(_rank_pos)

    # ── 12. MACD signal (12-26 EMA diff / std)
    def _macd_norm(s: pd.Series) -> pd.Series:
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd / (s.rolling(20).std() + 1e-10)

    out["tech_macd"] = grp["close"].transform(_macd_norm)

    # ── 13. 가격 가속도 (모멘텀 변화율)
    out["tech_accel"] = grp["close"].transform(
        lambda s: s.pct_change(5) - s.pct_change(5).shift(5)
    )

    return out


# 피처 이름 목록 (외부 참조용)
TECHNICAL_FEATURES = [
    "tech_mom1", "tech_mom5", "tech_mom10", "tech_mom20", "tech_mom60",
    "tech_ma5_dev", "tech_ma20_dev", "tech_ma60_dev",
    "tech_rsi14",
    "tech_bb_pos",
    "tech_vol20",
    "tech_vol_ratio", "tech_tv_ratio",
    "tech_hl_spread", "tech_close_pos",
    "tech_52w_pos",
    "tech_macd",
    "tech_accel",
]
