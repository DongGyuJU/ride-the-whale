"""
수급 피처 v2 — Fama-MacBeth 결과 기반 재설계

FM 결과 핵심 인사이트:
  - 장기 누적(20일)은 음의 신호 → 이미 오른 후 매수
  - 단기 가속도(fmom_chg)가 가장 강한 양의 신호
  - 당일 순위(fnet_rank)가 유효

신규 추가:
  - sup_fmom_accel: 외인 매수 가속도 강화 버전
  - sup_smart_money: 외인 단기 신호 종합 스코어
  - sup_divergence: 외인(양)-기관(음) 괴리 (기관 누적은 음의 신호)
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_supply_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(level="ticker", group_keys=False)
    date_grp = out.groupby(level="date", group_keys=False)

    has_f = "foreign_net" in out.columns
    has_i = "inst_net" in out.columns
    tv    = out["trade_value"].replace(0, np.nan)

    # ── 양(+) 방향 피처 ──────────────────────────────

    # 1. 외인 매수 가속도 (t=+3.26) — 핵심 신호
    #    최근 5일 누적 vs 이전 5~10일 누적
    if has_f:
        f5      = grp["foreign_net"].transform(lambda s: s.rolling(5).sum())
        f5_prev = grp["foreign_net"].transform(lambda s: s.rolling(5).sum().shift(5))
        tv10    = grp["trade_value"].transform(lambda s: s.rolling(10).sum())
        out["sup_fmom_chg"] = (f5 - f5_prev) / (tv10 + 1e-10)

        # 강화: 3일 vs 이전 3~6일 (더 단기)
        f3      = grp["foreign_net"].transform(lambda s: s.rolling(3).sum())
        f3_prev = grp["foreign_net"].transform(lambda s: s.rolling(3).sum().shift(3))
        tv6     = grp["trade_value"].transform(lambda s: s.rolling(6).sum())
        out["sup_fmom_accel"] = (f3 - f3_prev) / (tv6 + 1e-10)

    # 2. 당일 외인 순매수 시장 내 순위 (t=+2.34)
    if has_f:
        out["_fnet_norm_tmp"] = out["foreign_net"] / (tv + 1e-10)
        out["sup_fnet_rank"] = out.groupby(level="date")["_fnet_norm_tmp"].transform(
            lambda s: s.rank(pct=True, na_option="keep")
        )
        out.drop(columns=["_fnet_norm_tmp"], inplace=True)

    # 3. 외인 10일 누적 (t=+2.29) — 단기 유효
    if has_f:
        f10  = grp["foreign_net"].transform(lambda s: s.rolling(10).sum())
        tv10 = grp["trade_value"].transform(lambda s: s.rolling(10).sum())
        out["sup_fcum10"] = f10 / (tv10 + 1e-10)

    # 4. 기관 연속 매수 (t=+3.70 단독회귀) — 단기 유효
    if has_i:
        def _streak(s):
            result = pd.Series(0.0, index=s.index)
            streak = 0
            for i, v in enumerate(s):
                if pd.isna(v): streak = 0
                elif v > 0: streak = streak + 1 if streak >= 0 else 1
                elif v < 0: streak = streak - 1 if streak <= 0 else -1
                else: streak = 0
                result.iloc[i] = streak
            return result
        out["sup_istreak"] = grp["inst_net"].transform(_streak)

    # ── 음(-) 방향 피처 (역방향 신호 — 높으면 매도) ──

    # 5. 외인 20일 누적 (t=-7.09) — 장기 누적은 역방향
    if has_f:
        f20  = grp["foreign_net"].transform(lambda s: s.rolling(20).sum())
        tv20 = grp["trade_value"].transform(lambda s: s.rolling(20).sum())
        out["sup_fcum20"] = f20 / (tv20 + 1e-10)

    # 6. 기관 20일 누적 (t=-8.19) — 가장 강한 역방향 신호
    if has_i:
        i20  = grp["inst_net"].transform(lambda s: s.rolling(20).sum())
        tv20 = grp["trade_value"].transform(lambda s: s.rolling(20).sum())
        out["sup_icum20"] = i20 / (tv20 + 1e-10)

    # 7. 외인 연속 매수 (t=-4.52) — 연속 매수 = 이미 많이 올랐다
    if has_f:
        def _fstreak(s):
            result = pd.Series(0.0, index=s.index)
            streak = 0
            for i, v in enumerate(s):
                if pd.isna(v): streak = 0
                elif v > 0: streak = streak + 1 if streak >= 0 else 1
                elif v < 0: streak = streak - 1 if streak <= 0 else -1
                else: streak = 0
                result.iloc[i] = streak
            return result
        out["sup_fstreak"] = grp["foreign_net"].transform(_fstreak)

    # ── 신규: 외인-기관 괴리 신호 ──────────────────────
    # 외인 단기(양) - 기관 누적(음) → 괴리가 클수록 매수
    if has_f and has_i:
        if "sup_fcum10" in out.columns and "sup_icum20" in out.columns:
            # 외인 단기 모멘텀 - 기관 장기 누적 (역방향이라 음수로)
            out["sup_divergence"] = out["sup_fcum10"] - (-out["sup_icum20"])

    return out


SUPPLY_FEATURES_V2 = [
    # 양(+) 방향
    "sup_fmom_chg", "sup_fmom_accel", "sup_fnet_rank", "sup_fcum10", "sup_istreak",
    # 음(-) 방향
    "sup_fcum20", "sup_icum20", "sup_fstreak",
    # 신규
    "sup_divergence",
]
