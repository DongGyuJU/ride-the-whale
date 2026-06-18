"""
코랩 v7: 레짐별 독립 모델 + 레짐별 피처셋
=================================================

핵심 발견:
  강세장(A,B): 모멘텀/수급 76피처 최적
  약세장(C,D): 펀더멘털(공매도+어닝) 84피처 최적

레짐별 피처셋:
  A (강세+금리하락): 76피처 (IC 0.0967)
  B (강세+금리상승): 76피처 (IC 0.0616)
  C (약세+금리하락): 84피처 (IC 0.0116)
  D (약세+금리상승): 84피처 (IC 0.0606)
"""
from __future__ import annotations
import logging, pickle, sys, warnings
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConstantInputWarning)

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
MODEL_DIR = ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MARKET  = "KOSPI"
HORIZON = 20
FULL_START = "2018-01-01"
FULL_END   = "2024-12-31"

REGIMES = {
    "A": {"label": "강세+금리하락", "kospi_bull": True,  "rate_fall": True,
          "use_fundamental": False},  # 모멘텀 우세
    "B": {"label": "강세+금리상승", "kospi_bull": True,  "rate_fall": False,
          "use_fundamental": False},
    "C": {"label": "약세+금리하락", "kospi_bull": False, "rate_fall": True,
          "use_fundamental": True},   # 펀더멘털 우세
    "D": {"label": "약세+금리상승", "kospi_bull": False, "rate_fall": False,
          "use_fundamental": True},
}

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
EN_ALPHAS    = [0.001, 0.01, 0.1, 1.0]


def ic(y, p):
    mask = ~(np.isnan(y) | np.isnan(p))
    if mask.sum() < 10: return np.nan
    return spearmanr(y[mask], p[mask]).statistic


def add_interactions(feat_df, feat_cols):
    out = feat_df.copy(); new_cols = []
    rate = out["macro_rate_regime"] if "macro_rate_regime" in out.columns else None
    bull = out["macro_kospi_regime"] if "macro_kospi_regime" in out.columns else None
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


def get_regime_mask(macro_df, dates, regime_id):
    r = REGIMES[regime_id]
    macro_re = macro_df.reindex(dates.unique().sort_values()).ffill()
    kospi_bull = macro_re["macro_kospi_regime"] > 0.5
    rate_fall  = macro_re["macro_rate_regime"]  > 0.5
    if r["kospi_bull"] and r["rate_fall"]:
        regime_dates = set(macro_re[kospi_bull & rate_fall].index)
    elif r["kospi_bull"] and not r["rate_fall"]:
        regime_dates = set(macro_re[kospi_bull & ~rate_fall].index)
    elif not r["kospi_bull"] and r["rate_fall"]:
        regime_dates = set(macro_re[~kospi_bull & rate_fall].index)
    else:
        regime_dates = set(macro_re[~kospi_bull & ~rate_fall].index)
    return pd.DatetimeIndex(dates).map(lambda d: d in regime_dates).values


def build_features(macro_df, include_fundamental=False):
    """피처 빌드. include_fundamental=True면 공매도+어닝 포함."""
    panel = pd.read_parquet(RAW_DIR / f"{MARKET.lower()}_panel.parquet")
    univ  = ROOT/"results"/"diagnose"/f"universe_filter_{MARKET.lower()}.csv"
    universe = pd.read_csv(univ)["ticker"].astype(str).str.zfill(6).tolist() \
               if univ.exists() else None

    feat_df, feat_cols = build_features_v2(
        panel, mode="full", universe=universe, zscore=False, cache_path=None
    )
    panel_u = panel[panel.index.get_level_values("ticker").isin(
        feat_df.index.get_level_values("ticker").unique()
    )]

    # 매크로
    panel_m = add_macro_features(panel_u, macro_df)
    for col in MACRO_FEATURES:
        if col in panel_m.columns:
            feat_df[col] = panel_m[col]
    all_cols = feat_cols + [c for c in MACRO_FEATURES if c in feat_df.columns]

    # 교호작용
    feat_df, all_cols = add_interactions(feat_df, all_cols)

    # 미분
    feat_df, all_cols = add_derivative_features(feat_df, all_cols)

    sparse_cols = []

    # 펀더멘털 피처 (약세장에서만)
    if include_fundamental:
        if HAS_SHORT:
            short_df = load_short_data(SHORT_DIR)
            if short_df is not None:
                feat_df = add_short_features(feat_df, short_df)
                new_short = [c for c in SHORT_FEATURES if c in feat_df.columns]
                all_cols += new_short
                sparse_cols += new_short

        if HAS_EARN:
            earn_df = load_earnings_data(EARN_DIR)
            if earn_df is not None:
                feat_df = add_earnings_features(feat_df, earn_df)
                new_earn = [c for c in EARNINGS_FEATURES if c in feat_df.columns]
                all_cols += new_earn
                sparse_cols += new_earn

    # z-score
    feat_df = cross_sectional_zscore(feat_df, all_cols)

    # 레이블
    labels = winsorize_returns(make_forward_returns(panel_u, horizons=[HORIZON]))
    y_all  = labels[f"fwd{HORIZON}"]

    common = feat_df.index.intersection(y_all.index)
    X = feat_df.loc[common, all_cols].copy()
    y = y_all.loc[common]

    # 희소 피처 NaN → 0
    if sparse_cols:
        X[sparse_cols] = X[sparse_cols].fillna(0)

    core_cols = [c for c in all_cols if c not in sparse_cols]
    valid = X[core_cols].notna().all(axis=1) & y.notna()

    n_fund = len(sparse_cols)
    log.info(f"  피처: {len(all_cols)}개 (핵심 {len(core_cols)} + 희소 {n_fund}) | "
             f"유효: {valid.sum():,}행")
    return X[valid], y[valid], all_cols


def train_ensemble(X, y, feat_cols, label=""):
    scaler = StandardScaler()
    n = len(X)
    split = int(n * 0.8)

    X_tr = scaler.fit_transform(X.iloc[:split][feat_cols].fillna(0).values)
    y_tr = y.iloc[:split].values
    X_vl = scaler.transform(X.iloc[split:][feat_cols].fillna(0).values)
    y_vl = y.iloc[split:].values

    best_a, best_v = 0.1, -np.inf
    for a in RIDGE_ALPHAS:
        v = ic(y_vl, Ridge(alpha=a).fit(X_tr, y_tr).predict(X_vl))
        if not np.isnan(v) and v > best_v:
            best_v, best_a = v, a
    ridge = Ridge(alpha=best_a).fit(X_tr, y_tr)

    best_p, best_v2 = {"alpha":0.01,"l1_ratio":0.1}, -np.inf
    for l1 in EN_L1_RATIOS:
        for a in EN_ALPHAS:
            v = ic(y_vl, ElasticNet(alpha=a, l1_ratio=l1,
                   max_iter=2000, random_state=42).fit(X_tr, y_tr).predict(X_vl))
            if not np.isnan(v) and v > best_v2:
                best_v2, best_p = v, {"alpha":a,"l1_ratio":l1}
    en = ElasticNet(**best_p, max_iter=2000, random_state=42).fit(X_tr, y_tr)

    ens_val = ic(y_vl, 0.5*ridge.predict(X_vl) + 0.5*en.predict(X_vl))
    log.info(f"  [{label}] Ridge={best_v:.4f} EN={best_v2:.4f} Ens={ens_val:.4f}")

    X_all = scaler.transform(X[feat_cols].fillna(0).values)
    return {
        "ridge":     Ridge(alpha=best_a).fit(X_all, y.values),
        "en":        ElasticNet(**best_p, max_iter=2000, random_state=42).fit(X_all, y.values),
        "scaler":    scaler,
        "feat_cols": feat_cols,
        "val_ic":    ens_val,
        "n_samples": n,
    }


if __name__ == "__main__":
    log.info("매크로 데이터 로드...")
    macro_df = load_macro(MACRO_DIR)

    # 피처셋 2종 빌드
    log.info("\n피처셋 1: 모멘텀 (강세장용, 76피처)")
    X_mom, y_mom, cols_mom = build_features(macro_df, include_fundamental=False)

    log.info("\n피처셋 2: 펀더멘털 (약세장용, 84피처)")
    X_fund, y_fund, cols_fund = build_features(macro_df, include_fundamental=True)

    # 기간 필터
    def filter_period(X, y):
        dates = X.index.get_level_values("date")
        mask  = (dates >= FULL_START) & (dates <= FULL_END)
        return X[mask], y[mask]

    X_mom,  y_mom  = filter_period(X_mom,  y_mom)
    X_fund, y_fund = filter_period(X_fund, y_fund)

    # 레짐별 학습
    regime_models = {}
    regime_stats  = {}

    log.info(f"\n{'='*60}")
    log.info("  레짐별 독립 모델 학습 (레짐별 최적 피처셋)")
    log.info(f"{'='*60}")

    for regime_id, info in REGIMES.items():
        use_fund = info["use_fundamental"]
        X, y, cols = (X_fund, y_fund, cols_fund) if use_fund else (X_mom, y_mom, cols_mom)
        feat_label = "84피처(펀더멘털)" if use_fund else "76피처(모멘텀)"

        dates  = X.index.get_level_values("date")
        r_mask = get_regime_mask(macro_df, dates, regime_id)
        X_r, y_r = X[r_mask], y[r_mask]
        n_r = len(X_r)
        pct = n_r / len(X) * 100

        log.info(f"\n  레짐 {regime_id}: {info['label']} ({feat_label})")
        log.info(f"  데이터: {n_r:,}행 ({pct:.1f}%)")

        if n_r < 1000:
            log.warning(f"  데이터 부족 → 스킵")
            regime_models[regime_id] = None
            regime_stats[regime_id]  = {"n": n_r, "val_ic": None, "feat_label": feat_label}
            continue

        model = train_ensemble(X_r, y_r, cols,
                               label=f"{regime_id}:{info['label']}")
        regime_models[regime_id] = model
        regime_stats[regime_id]  = {
            "n": n_r, "pct": pct,
            "val_ic": model["val_ic"],
            "feat_label": feat_label,
        }

    # 결과 비교
    log.info(f"\n{'='*65}")
    log.info("  최종 결과 비교")
    log.info(f"{'='*65}")
    log.info(f"  {'레짐':<3} {'설명':<14} {'피처셋':<16} {'데이터':>8} {'val IC':>8}")
    log.info(f"  {'-'*55}")

    prev_v7 = {"A":0.0967,"B":0.0616,"C":0.0069,"D":0.0596}
    for rid, stats in regime_stats.items():
        ic_val = stats.get("val_ic")
        ic_str = f"{ic_val:+.4f}" if ic_val else "부족"
        prev   = prev_v7.get(rid, 0)
        arrow  = "↑" if (ic_val or 0) > prev else "↓" if (ic_val or 0) < prev else "→"
        log.info(f"  {rid:<3} {REGIMES[rid]['label']:<14} "
                 f"{stats['feat_label']:<16} "
                 f"{stats['n']:>8,} {ic_str:>8} {arrow} (이전: {prev:.4f})")

    # 저장
    save_obj = {
        "type":         "regime_ensemble_v7",
        "regimes":      regime_models,
        "regime_defs":  REGIMES,
        "feat_cols":    cols_mom,       # 기본 (레짐 A/B용)
        "feat_cols_fund": cols_fund,    # 펀더멘털 (레짐 C/D용)
        "macro_features": MACRO_FEATURES,
        "stats":        regime_stats,
    }

    save_path = MODEL_DIR / f"regime_v7_{MARKET.lower()}_fwd{HORIZON}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(save_obj, f)
    log.info(f"\n  💾 저장: {save_path}")
    log.info(f"\n  v6 단일(0.0376) 대비 v7 최고 레짐A: "
             f"{regime_stats['A'].get('val_ic',0):+.4f}")
