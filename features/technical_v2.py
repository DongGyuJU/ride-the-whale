"""
기술적 피처 v2 — Fama-MacBeth 유의 피처만 (8개)

FM 결과 기반:
  양(+): tech_52w_pos (t=14.40), tech_bb_pos (t=3.00),
          tech_tv_ratio (t=2.87), tech_macd (t=2.71), tech_mom20 (t=2.60)
  음(-): tech_vol20 (t=-11.31), tech_mom60 (t=-5.04), tech_vol_ratio (t=-3.03)
         tech_hl_spread (t=-2.24), tech_ma20_dev (t=-2.45), tech_ma5_dev (t=-2.12)

제거된 피처 (|t| < 2):
  tech_mom1, tech_mom5, tech_mom10, tech_rsi14, tech_accel,
  tech_ma60_dev, tech_close_pos, tech_macd (스케일 이상)
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_technical_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(level="ticker", group_keys=False)

    close  = out["close"]
    high   = out["high"]
    low    = out["low"]
    volume = out["volume"]

    # ── 양(+) 방향 피처 ──────────────────────────────

    # 1. 52주 고저 대비 위치 (t=+14.40) — 모멘텀 핵심
    def _rank_pos(s, n=252):
        mn = s.rolling(n, min_periods=60).min()
        mx = s.rolling(n, min_periods=60).max()
        return (s - mn) / (mx - mn + 1e-10)
    out["tech_52w_pos"] = grp["close"].transform(_rank_pos)

    # 2. 볼린저 밴드 위치 (t=+3.00)
    def _bb_pos(s, n=20):
        ma  = s.rolling(n).mean()
        std = s.rolling(n).std()
        return (s - (ma - 2*std)) / (4*std + 1e-10)
    out["tech_bb_pos"] = grp["close"].transform(_bb_pos)

    # 3. 거래대금 모멘텀 (t=+2.87)
    out["tech_tv_ratio"] = grp["trade_value"].transform(
        lambda s: s / (s.rolling(20).mean() + 1e-10)
    )

    # 4. MACD 정규화 (t=+2.71) — 스케일 문제 해결: 수익률 단위로
    def _macd_norm(s):
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd  = (ema12 - ema26) / (s + 1e-10)  # 가격 대비 비율로 정규화
        return macd
    out["tech_macd"] = grp["close"].transform(_macd_norm)

    # 5. 20일 모멘텀 (t=+2.60)
    out["tech_mom20"] = grp["close"].transform(lambda s: s.pct_change(20))

    # ── 음(-) 방향 피처 (높을수록 수익 낮음) ──────────

    # 6. 20일 변동성 (t=-11.31)
    out["tech_vol20"] = grp["close"].transform(
        lambda s: s.pct_change().rolling(20).std()
    )

    # 7. 60일 모멘텀 (t=-5.04) — 장기 과매수 역전
    out["tech_mom60"] = grp["close"].transform(lambda s: s.pct_change(60))

    # 8. 거래량 모멘텀 (t=-3.03) — 과다 거래 = 단기 고점
    out["tech_vol_ratio"] = grp["volume"].transform(
        lambda s: s / (s.rolling(20).mean() + 1e-10)
    )

    # 9. 고저 스프레드 (t=-2.24)
    out["tech_hl_spread"] = (high - low) / (close + 1e-10)

    # 10. 20일 MA 괴리율 (t=-2.45)
    ma20 = grp["close"].transform(lambda s: s.rolling(20).mean())
    out["tech_ma20_dev"] = (close - ma20) / (ma20 + 1e-10)

    # 11. 5일 MA 괴리율 (t=-2.12)
    ma5 = grp["close"].transform(lambda s: s.rolling(5).mean())
    out["tech_ma5_dev"] = (close - ma5) / (ma5 + 1e-10)

    return out


TECHNICAL_FEATURES_V2 = [
    # 양(+)
    "tech_52w_pos", "tech_bb_pos", "tech_tv_ratio", "tech_macd", "tech_mom20",
    # 음(-)
    "tech_vol20", "tech_mom60", "tech_vol_ratio", "tech_hl_spread",
    "tech_ma20_dev", "tech_ma5_dev",
]
