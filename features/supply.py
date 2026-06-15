"""
수급 피처 (외인/기관 매매 기반, ~20개).

핵심 아이디어:
- 누적 순매수 (5d/10d/20d): 추세 파악
- 거래대금 대비 정규화: 종목 간 비교 가능하게
- Concordance (외인+기관 합의): 동반 매수 = 강한 시그널
- 연속 매수 streak: 지속성 측정
- Cross-sectional rank: 당일 시장 내 상대 강도

모든 피처는 causal (미래 누설 없음).
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_supply_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: MultiIndex (date, ticker) 패널.
            필수 컬럼: foreign_net, inst_net, retail_net, trade_value

    Returns:
        수급 피처 컬럼이 추가된 DataFrame.
        새 컬럼 prefix: 'sup_'
    """
    out = df.copy()
    grp = out.groupby(level="ticker", group_keys=False)

    has_foreign = "foreign_net" in out.columns
    has_inst = "inst_net" in out.columns
    tv = out["trade_value"].replace(0, np.nan)

    # ─────────────────────────────────────────
    # 1. 거래대금 대비 정규화 순매수
    # ─────────────────────────────────────────
    if has_foreign:
        out["sup_fnet_norm"] = out["foreign_net"] / (tv + 1e-10)
    if has_inst:
        out["sup_inet_norm"] = out["inst_net"] / (tv + 1e-10)

    # ─────────────────────────────────────────
    # 2. 누적 순매수 (rolling sum, 정규화)
    # ─────────────────────────────────────────
    for w in [5, 10, 20]:
        if has_foreign:
            cum = grp["foreign_net"].transform(lambda s: s.rolling(w).sum())
            cum_tv = grp["trade_value"].transform(lambda s: s.rolling(w).sum())
            out[f"sup_fcum{w}"] = cum / (cum_tv + 1e-10)

        if has_inst:
            cum = grp["inst_net"].transform(lambda s: s.rolling(w).sum())
            cum_tv = grp["trade_value"].transform(lambda s: s.rolling(w).sum())
            out[f"sup_icum{w}"] = cum / (cum_tv + 1e-10)

    # ─────────────────────────────────────────
    # 3. Concordance — 외인 + 기관 동반 매수
    # ─────────────────────────────────────────
    if has_foreign and has_inst:
        f_sign = np.sign(out["foreign_net"].fillna(0))
        i_sign = np.sign(out["inst_net"].fillna(0))
        out["sup_concordance"] = (f_sign * i_sign)  # +1: 동방향, -1: 역방향, 0: 한쪽 0

        # 5일/10일 누적 concordance (동반 매수 일수 비율)
        for w in [5, 10]:
            out[f"sup_conc{w}"] = grp["sup_concordance"].transform(
                lambda s: s.rolling(w).mean()
            )

    # ─────────────────────────────────────────
    # 4. 연속 매수 streak
    # ─────────────────────────────────────────
    def _streak(s: pd.Series) -> pd.Series:
        """연속 양수 일수 (음수면 리셋). 매도면 음수 streak."""
        result = pd.Series(0.0, index=s.index)
        streak = 0
        for i, v in enumerate(s):
            if pd.isna(v):
                streak = 0
            elif v > 0:
                streak = streak + 1 if streak >= 0 else 1
            elif v < 0:
                streak = streak - 1 if streak <= 0 else -1
            else:
                streak = 0
            result.iloc[i] = streak
        return result

    if has_foreign:
        out["sup_fstreak"] = grp["foreign_net"].transform(_streak)
    if has_inst:
        out["sup_istreak"] = grp["inst_net"].transform(_streak)

    # ─────────────────────────────────────────
    # 5. 수급 모멘텀 변화 (최근 5일 vs 이전 5일)
    # ─────────────────────────────────────────
    if has_foreign:
        f5 = grp["foreign_net"].transform(lambda s: s.rolling(5).sum())
        f5_prev = f5.groupby(level="ticker").shift(5)
        cum_tv5 = grp["trade_value"].transform(lambda s: s.rolling(5).sum())
        out["sup_fmom_chg"] = (f5 - f5_prev) / (cum_tv5 + 1e-10)

    # ─────────────────────────────────────────
    # 6. Cross-sectional rank (당일 시장 내 상대 위치)
    # 주의: 이 피처는 date 레벨 groupby — 미래 누설 없음
    # ─────────────────────────────────────────
    date_grp = out.groupby(level="date", group_keys=False)

    if has_foreign:
        out["sup_fnet_rank"] = date_grp["sup_fnet_norm"].transform(
            lambda s: s.rank(pct=True, na_option="keep")
        )
    if has_inst:
        out["sup_inet_rank"] = date_grp["sup_inet_norm"].transform(
            lambda s: s.rank(pct=True, na_option="keep")
        )

    if "sup_fcum20" in out.columns:
        out["sup_fcum20_rank"] = date_grp["sup_fcum20"].transform(
            lambda s: s.rank(pct=True, na_option="keep")
        )

    return out


# 피처 이름 목록
SUPPLY_FEATURES = [
    "sup_fnet_norm", "sup_inet_norm",
    "sup_fcum5", "sup_fcum10", "sup_fcum20",
    "sup_icum5", "sup_icum10", "sup_icum20",
    "sup_concordance", "sup_conc5", "sup_conc10",
    "sup_fstreak", "sup_istreak",
    "sup_fmom_chg",
    "sup_fnet_rank", "sup_inet_rank", "sup_fcum20_rank",
]
