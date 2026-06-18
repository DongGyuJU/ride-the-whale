"""
Fama-MacBeth 재검증 v2 — 84피처 기준
======================================

기존 76피처 + 공매도(4) + 어닝(4) = 84피처 전체 t-stat 검증.

핵심 질문:
  short_squeeze, earn_accel이 진짜 유의미한 신호인가?
  → |t| > 2 이면 유효 피처
  → |t| < 2 이면 노이즈 → 제거 검토

출력:
  results/fm_v2_all.csv      — 전체 84피처 t-stat
  results/fm_v2_short.csv    — 유의 피처만 (|t|>2)
  results/fm_v2_regime.csv   — 레짐별 t-stat 비교
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from features.pipeline_v2 import build_features_v2, cross_sectional_zscore
from features.macro import load_macro, add_macro_features, MACRO_FEATURES
from features.derivatives import add_derivative_features
from labels.forward_return import make_forward_returns, winsorize_returns

try:
    from features.short_selling import load_short_data, add_short_features, SHORT_FEATURES
    HAS_SHORT = True
except: HAS_SHORT = False

try:
    from features.earnings import load_earnings_data, add_earnings_features, EARNINGS_FEATURES
    HAS_EARN = True
except: HAS_EARN = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR   = ROOT / "data" / "raw"
MACRO_DIR = ROOT / "data" / "macro"
SHORT_DIR = ROOT / "data" / "short"
EARN_DIR  = ROOT / "data" / "earnings"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

MARKET  = "KOSPI"
HORIZON = 20
START   = "2018-01-01"
END     = "2024-12-31"


def add_interactions(feat_df, feat_cols):
    out = feat_df.copy(); new_cols = []
    rate = out.get("macro_rate_regime")
    bull = out.get("macro_kospi_regime")
    targets = ["tech_52w_pos","tech_bb_pos","tech_tv_ratio","tech_mom20",
               "tech_vol20","tech_mom60","sup_fmom_chg","sup_fnet_rank",
               "sup_fcum20","sup_fstreak"]
    for t in targets:
        if t not in out.columns: continue
        if rate is not None:
            c = f"ix_rate_{t}"; out[c] = rate * out[t]; new_cols.append(c)
        if bull is not None:
            c = f"ix_bull_{t}"; out[c] = bull * out[t]; new_cols.append(c)
    return out, feat_cols + new_cols


def fama_macbeth(X: pd.DataFrame, y: pd.Series, feat_cols: list) -> pd.DataFrame:
    """Fama-MacBeth 회귀: 날짜별 단변량 IC → 평균/t-stat."""
    dates = X.index.get_level_values("date").unique()
    betas = {col: [] for col in feat_cols}

    for d in dates:
        mask = X.index.get_level_values("date") == d
        X_d  = X[mask]
        y_d  = y[mask]
        valid = y_d.notna()
        if valid.sum() < 30:
            continue
        for col in feat_cols:
            if col not in X_d.columns:
                betas[col].append(np.nan)
                continue
            x = X_d[col][valid]
            yv = y_d[valid]
            both = x.notna() & yv.notna()
            if both.sum() < 20:
                betas[col].append(np.nan)
                continue
            ic = spearmanr(x[both], yv[both]).statistic
            betas[col].append(ic)

    rows = []
    for col, vals in betas.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) < 10:
            continue
        mean = arr.mean()
        std  = arr.std(ddof=1)
        n    = len(arr)
        t    = mean / (std / np.sqrt(n) + 1e-10)
        rows.append({
            "feature": col,
            "mean_ic": round(mean, 5),
            "std_ic":  round(std, 5),
            "t_stat":  round(t, 2),
            "n_days":  n,
            "sig":     "✅" if abs(t) > 2 else "⚠️" if abs(t) > 1.5 else "❌",
        })

    return pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False)


def main():
    log.info("매크로 데이터 로드...")
    macro_df = load_macro(MACRO_DIR)

    log.info("피처 빌드...")
    panel = pd.read_parquet(RAW_DIR / f"{MARKET.lower()}_panel.parquet")
    univ  = ROOT/"results"/"diagnose"/f"universe_filter_{MARKET.lower()}.csv"
    universe = pd.read_csv(univ)["ticker"].astype(str).str.zfill(6).tolist()

    feat_df, feat_cols = build_features_v2(
        panel, mode="full", universe=universe, zscore=False, cache_path=None
    )
    panel_u = panel[panel.index.get_level_values("ticker").isin(
        feat_df.index.get_level_values("ticker").unique()
    )]

    # 매크로 + 교호작용 + 미분
    panel_m = add_macro_features(panel_u, macro_df)
    for col in MACRO_FEATURES:
        if col in panel_m.columns:
            feat_df[col] = panel_m[col]
    all_cols = feat_cols + [c for c in MACRO_FEATURES if c in feat_df.columns]
    feat_df, all_cols = add_interactions(feat_df, all_cols)
    feat_df, all_cols = add_derivative_features(feat_df, all_cols)

    # 공매도
    sparse = []
    if HAS_SHORT:
        short_df = load_short_data(SHORT_DIR)
        if short_df is not None:
            feat_df = add_short_features(feat_df, short_df)
            new_s = [c for c in SHORT_FEATURES if c in feat_df.columns]
            all_cols += new_s; sparse += new_s

    # 어닝
    if HAS_EARN:
        earn_df = load_earnings_data(EARN_DIR)
        if earn_df is not None:
            feat_df = add_earnings_features(feat_df, earn_df)
            new_e = [c for c in EARNINGS_FEATURES if c in feat_df.columns]
            all_cols += new_e; sparse += new_e

    # Z-score
    feat_df = cross_sectional_zscore(feat_df, all_cols)
    if sparse:
        feat_df[sparse] = feat_df[sparse].fillna(0)

    # 레이블
    labels = winsorize_returns(make_forward_returns(panel_u, horizons=[HORIZON]))
    y_all  = labels[f"fwd{HORIZON}"]

    common = feat_df.index.intersection(y_all.index)
    X = feat_df.loc[common, all_cols].copy()
    y = y_all.loc[common]

    # 기간 필터
    dates = X.index.get_level_values("date")
    mask  = (dates >= START) & (dates <= END)
    X, y  = X[mask], y[mask]
    dates = X.index.get_level_values("date")

    log.info(f"  데이터: {len(X):,}행 | 피처: {len(all_cols)}개")

    # ── 전체 FM ──
    log.info("\n전체 84피처 FM 검증...")
    fm_all = fama_macbeth(X, y, all_cols)

    fm_all.to_csv(RESULT_DIR/"fm_v2_all.csv", index=False)
    fm_sig = fm_all[fm_all["t_stat"].abs() > 2]
    fm_sig.to_csv(RESULT_DIR/"fm_v2_significant.csv", index=False)

    # ── 레짐별 FM ──
    log.info("\n레짐별 FM 검증...")
    regime_rows = []
    macro_re = macro_df.reindex(
        pd.DatetimeIndex(dates.unique()).sort_values()
    ).ffill()

    for regime_id, (bull, fall) in {
        "A": (True,  True),
        "B": (True,  False),
        "C": (False, True),
        "D": (False, False),
    }.items():
        kospi_bull = macro_re["macro_kospi_regime"] > 0.5
        rate_fall  = macro_re["macro_rate_regime"]  > 0.5
        if bull and fall:
            rdates = set(macro_re[kospi_bull & rate_fall].index)
        elif bull:
            rdates = set(macro_re[kospi_bull & ~rate_fall].index)
        elif fall:
            rdates = set(macro_re[~kospi_bull & rate_fall].index)
        else:
            rdates = set(macro_re[~kospi_bull & ~rate_fall].index)

        r_mask = pd.DatetimeIndex(dates).map(lambda d: d in rdates).values
        X_r, y_r = X[r_mask], y[r_mask]

        # 핵심 피처 + 공매도/어닝만
        key_feats = [
            "tech_52w_pos","sup_fmom_chg","sup_fcum20","ix_bull_tech_52w_pos",
            "d5_macro_rate10_chg20",
        ] + sparse

        fm_r = fama_macbeth(X_r, y_r, key_feats)
        for _, row in fm_r.iterrows():
            regime_rows.append({"regime": regime_id, **row.to_dict()})

    fm_regime = pd.DataFrame(regime_rows)
    fm_regime.to_csv(RESULT_DIR/"fm_v2_regime.csv", index=False)

    # ── 결과 출력 ──
    log.info(f"\n{'='*65}")
    log.info("  전체 피처 FM 결과 (상위 20개, |t-stat| 기준)")
    log.info(f"{'='*65}")
    log.info(f"  {'피처':<35} {'mean IC':>8} {'t-stat':>8} {'sig':>4}")
    log.info(f"  {'-'*60}")
    for _, r in fm_all.head(20).iterrows():
        log.info(f"  {r['feature']:<35} {r['mean_ic']:>+8.5f} "
                 f"{r['t_stat']:>8.2f} {r['sig']:>4}")

    log.info(f"\n  유의 피처 수 (|t|>2): {len(fm_sig)}개 / {len(fm_all)}개")

    # 공매도/어닝 피처 별도 출력
    log.info(f"\n{'='*65}")
    log.info("  공매도 + 어닝 피처 검증")
    log.info(f"{'='*65}")
    new_feats = fm_all[fm_all["feature"].str.startswith(("short_","earn_"))]
    log.info(f"  {'피처':<35} {'mean IC':>8} {'t-stat':>8} {'sig':>4}")
    log.info(f"  {'-'*55}")
    for _, r in new_feats.iterrows():
        log.info(f"  {r['feature']:<35} {r['mean_ic']:>+8.5f} "
                 f"{r['t_stat']:>8.2f} {r['sig']:>4}")

    log.info(f"\n  📁 저장:")
    log.info(f"     results/fm_v2_all.csv")
    log.info(f"     results/fm_v2_significant.csv")
    log.info(f"     results/fm_v2_regime.csv")


if __name__ == "__main__":
    main()
