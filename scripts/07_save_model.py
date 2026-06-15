"""
Stage 6-pre: 학습된 Ridge 모델 저장
======================================
실행:
  python3 scripts/07_save_model.py [--market KOSPI] [--horizon 20] [--mode full]
"""
from __future__ import annotations
import argparse, logging, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from features.pipeline import build_features
from labels.forward_return import make_forward_returns, winsorize_returns
from models.temporal_split import temporal_split

RAW_DIR   = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "cache" / "features"
MODEL_DIR = ROOT / "models" / "saved"
HORIZONS  = [5, 10, 20]

# OOM 방지 — 최근 N rows만 학습
MAX_TRAIN_ROWS = 150_000
ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 1000.0]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])


def spearman_ic(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    return spearmanr(y_true[mask], y_pred[mask]).statistic


def subsample(X, y, max_rows):
    if len(X) <= max_rows:
        return X, y
    # 최근 우선
    idx = np.argsort(X.index.get_level_values("date"))[::-1][:max_rows]
    idx = np.sort(idx)
    logging.getLogger(__name__).info(f"  서브샘플: {len(X):,} → {max_rows:,} rows")
    return X.iloc[idx], y.iloc[idx]


def train_and_save(market: str, mode: str, horizon: int) -> None:
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*50}")
    log.info(f"  [{market} {mode} fwd{horizon}] 학습 + 저장")
    log.info(f"{'='*50}")

    panel = pd.read_parquet(RAW_DIR / f"{market.lower()}_panel.parquet")

    univ_path = ROOT / "results" / "diagnose" / f"universe_filter_{market.lower()}.csv"
    universe = pd.read_csv(univ_path)["ticker"].tolist() if univ_path.exists() else None

    cache_path = str(CACHE_DIR / f"{market.lower()}_{mode}.parquet")
    feat_df, feat_cols = build_features(
        panel, mode=mode, universe=universe,
        zscore=True, cache_path=cache_path,
    )

    tickers = feat_df.index.get_level_values("ticker").unique()
    panel_uni = panel[panel.index.get_level_values("ticker").isin(tickers)]
    labels = make_forward_returns(panel_uni, horizons=[horizon])
    labels = winsorize_returns(labels)
    y_all = labels[f"fwd{horizon}"]

    common = feat_df.index.intersection(y_all.index)
    X_all  = feat_df.loc[common, feat_cols]
    y_all  = y_all.loc[common]
    valid  = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[valid], y_all[valid]

    splits = temporal_split(X_all, y_all.to_frame())
    X_tr = splits["train"][0];  y_tr = splits["train"][1].iloc[:, 0]
    X_vl = splits["val"][0];    y_vl = splits["val"][1].iloc[:, 0]

    # 서브샘플
    X_tr, y_tr = subsample(X_tr, y_tr, MAX_TRAIN_ROWS)

    # 스케일러
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr[feat_cols].fillna(0).values)
    X_vl_s = scaler.transform(X_vl[feat_cols].fillna(0).values)
    y_tr_v = y_tr.values
    y_vl_v = y_vl.values

    # alpha 탐색
    best_alpha, best_ic = 10.0, -np.inf
    for alpha in ALPHA_CANDIDATES:
        m = Ridge(alpha=alpha)
        m.fit(X_tr_s, y_tr_v)
        ic = spearman_ic(y_vl_v, m.predict(X_vl_s))
        log.info(f"  alpha={alpha:>8.1f}  val IC={ic:.4f}")
        if not np.isnan(ic) and ic > best_ic:
            best_ic, best_alpha = ic, alpha

    log.info(f"  → 최적 alpha={best_alpha}  val IC={best_ic:.4f}")

    # train+val 합쳐 최종 학습
    X_vl_s2 = scaler.transform(X_vl[feat_cols].fillna(0).values)
    X_full = np.vstack([X_tr_s, X_vl_s2])
    y_full = np.concatenate([y_tr_v, y_vl_v])
    final = Ridge(alpha=best_alpha)
    final.fit(X_full, y_full)

    # 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"ridge_{market.lower()}_{mode}_fwd{horizon}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": final, "scaler": scaler, "feat_cols": feat_cols}, f)
    log.info(f"  ✅ 저장 완료: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market",  choices=["KOSPI","KOSDAQ","ALL"], default="KOSPI")
    parser.add_argument("--horizon", choices=["5","10","20","ALL"],    default="20")
    parser.add_argument("--mode",    choices=["ohlcv","full","both"],  default="full")
    args = parser.parse_args()

    setup_logging()
    markets  = ["KOSPI","KOSDAQ"] if args.market == "ALL" else [args.market]
    horizons = HORIZONS if args.horizon == "ALL" else [int(args.horizon)]
    modes    = ["ohlcv","full"] if args.mode == "both" else [args.mode]

    for market in markets:
        for mode in modes:
            for h in horizons:
                try:
                    train_and_save(market, mode, h)
                except Exception as e:
                    logging.getLogger(__name__).exception(f"  ❌ {e}")

    logging.getLogger(__name__).info(
        "\n다음 단계: python3 scripts/07_daily_signal.py --market KOSPI"
    )

if __name__ == "__main__":
    main()
