"""
Stage 3: LightGBM 학습 + 피처 중요도 분석
============================================

실험:
  A. KOSPI  OHLCV only
  B. KOSPI  OHLCV + 수급

Ridge 결과와 비교:
  - LGB가 Ridge를 이기는가?
  - 어떤 수급 피처가 진짜 알파인가? (gain importance)

실행:
  python3 scripts/04_train_lgb.py [--market KOSPI|KOSDAQ|ALL] [--horizon 5|10|20|ALL]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from features.pipeline import build_features
from labels.forward_return import make_forward_returns, winsorize_returns
from models.temporal_split import temporal_split, SPLIT
from models.lgb import train_lgb, predict_lgb, get_feature_importance
from evaluate.ic import compute_daily_ic, summarize_ic
from evaluate.long_short import compute_long_short_returns, summarize_long_short

RAW_DIR     = ROOT / "data" / "raw"
CACHE_DIR   = ROOT / "cache" / "features"
RESULTS_DIR = ROOT / "results" / "lgb"
HORIZONS    = [5, 10, 20]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "04_train_lgb.log", mode="a"),
        ],
    )


def load_universe(market: str) -> list[str] | None:
    path = ROOT / "results" / "diagnose" / f"universe_filter_{market.lower()}.csv"
    if path.exists():
        return pd.read_csv(path)["ticker"].tolist()
    return None


def run_experiment(
    market: str,
    mode: str,
    horizon: int,
    panel: pd.DataFrame,
    universe: list[str] | None,
) -> dict:
    log = logging.getLogger(__name__)
    exp_name = f"{market}_lgb_{mode}_fwd{horizon}"
    log.info(f"\n{'='*60}")
    log.info(f"  실험: {exp_name}")
    log.info(f"{'='*60}")

    # 1. 피처 (Ridge에서 캐싱된 것 재사용)
    cache_path = str(CACHE_DIR / f"{market.lower()}_{mode}.parquet")
    feat_df, feat_cols = build_features(
        panel,
        mode=mode,
        universe=universe,
        zscore=True,
        cache_path=cache_path,
    )
    if not feat_cols:
        log.error("  피처 없음 — 스킵")
        return {}

    # 2. 레이블
    tickers = feat_df.index.get_level_values("ticker").unique()
    panel_uni = panel[panel.index.get_level_values("ticker").isin(tickers)]
    labels = make_forward_returns(panel_uni, horizons=[horizon])
    labels = winsorize_returns(labels)
    y_all = labels[f"fwd{horizon}"]

    # 3. 정렬 + NaN 제거
    common = feat_df.index.intersection(y_all.index)
    X_all = feat_df.loc[common, feat_cols]
    y_all = y_all.loc[common]
    valid = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[valid], y_all[valid]

    log.info(f"  유효 데이터: {len(X_all):,} rows, {len(feat_cols)} 피처")

    # 4. 분할
    splits = temporal_split(X_all, y_all.to_frame())
    X_tr, y_tr_df = splits["train"]
    X_vl, y_vl_df = splits["val"]
    X_te, y_te_df = splits["test"]
    y_tr = y_tr_df.iloc[:, 0]
    y_vl = y_vl_df.iloc[:, 0]
    y_te = y_te_df.iloc[:, 0]

    # 5. 학습
    log.info("\n  LightGBM 학습 중 (early stopping)...")
    booster, val_ic = train_lgb(X_tr, y_tr, X_vl, y_vl, feat_cols)

    # 6. Test 평가
    pred_te = predict_lgb(booster, X_te, feat_cols)
    daily_ic = compute_daily_ic(pred_te, y_te)
    ic_stats = summarize_ic(daily_ic, label=exp_name)

    ls_df = compute_long_short_returns(pred_te, y_te, q=0.1)
    ls_stats = summarize_long_short(ls_df, label=exp_name)

    # 7. 피처 중요도
    imp = get_feature_importance(booster, feat_cols, importance_type="gain")
    log.info(f"\n  Top-15 피처 (gain importance):")
    for fname, val in imp.head(15).items():
        tag = " ◀ 수급" if fname.startswith("sup_") else ""
        log.info(f"    {fname:<30} {val:>10.1f}{tag}")

    # 수급 피처 중요도 합계
    sup_imp = imp[imp.index.str.startswith("sup_")]
    tech_imp = imp[imp.index.str.startswith("tech_")]
    total_imp = imp.sum()
    if total_imp > 0:
        log.info(f"\n  수급 피처 중요도 비중: {sup_imp.sum()/total_imp:.1%}")
        log.info(f"  기술적 피처 중요도 비중: {tech_imp.sum()/total_imp:.1%}")

    # 8. 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    daily_ic.to_csv(RESULTS_DIR / f"daily_ic_{exp_name}.csv")
    ls_df.to_csv(RESULTS_DIR / f"ls_{exp_name}.csv")
    imp.to_csv(RESULTS_DIR / f"importance_{exp_name}.csv")

    result = {
        "experiment": exp_name,
        "market": market,
        "mode": mode,
        "horizon": horizon,
        "n_features": len(feat_cols),
        "val_ic": val_ic,
        **ic_stats,
        **{f"ls_{k}": v for k, v in ls_stats.items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 학습")
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ", "ALL"], default="KOSPI")
    parser.add_argument("--horizon", choices=["5", "10", "20", "ALL"], default="ALL")
    args = parser.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    setup_logging()
    log = logging.getLogger(__name__)

    markets = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]
    horizons = HORIZONS if args.horizon == "ALL" else [int(args.horizon)]
    modes = ["ohlcv", "full"]

    all_results = []
    t0 = time.time()

    for market in markets:
        panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
        if not panel_path.exists():
            log.error(f"  ❌ {market} 패널 없음.")
            continue

        log.info(f"\n{'#'*60}")
        log.info(f"  {market} 패널 로드 중...")
        panel = pd.read_parquet(panel_path)
        universe = load_universe(market)

        for mode in modes:
            for horizon in horizons:
                try:
                    result = run_experiment(market, mode, horizon, panel, universe)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    log.exception(f"  ❌ {market}/{mode}/fwd{horizon} 실패: {e}")

    # ── Ridge vs LGB 비교
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = RESULTS_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        log.info(f"\n{'='*60}")
        log.info("  LGB 최종 결과")
        log.info(f"{'='*60}")
        cols = ["experiment", "IC_mean", "IC_IR", "IC_t", "ls_sharpe"]
        avail = [c for c in cols if c in summary_df.columns]
        log.info("\n" + summary_df[avail].to_string(index=False))

        # Ridge 결과 불러와서 비교
        ridge_path = ROOT / "results" / "ridge" / "summary.csv"
        if ridge_path.exists():
            ridge_df = pd.read_csv(ridge_path)
            log.info(f"\n{'='*60}")
            log.info("  Ridge vs LGB 비교 (IC_mean, test set)")
            log.info(f"{'='*60}")
            for market in markets:
                for mode in modes:
                    for h in horizons:
                        r_row = ridge_df[
                            (ridge_df["market"] == market) &
                            (ridge_df["mode"] == mode) &
                            (ridge_df["horizon"] == h)
                        ]
                        l_row = summary_df[
                            (summary_df["market"] == market) &
                            (summary_df["mode"] == mode) &
                            (summary_df["horizon"] == h)
                        ]
                        if r_row.empty or l_row.empty:
                            continue
                        r_ic = r_row["IC_mean"].values[0]
                        l_ic = l_row["IC_mean"].values[0]
                        winner = "LGB ✅" if l_ic > r_ic else "Ridge ✅"
                        log.info(
                            f"  [{market} {mode} fwd{h}]  "
                            f"Ridge {r_ic:.4f} vs LGB {l_ic:.4f}  → {winner}"
                        )

        log.info(f"\n  결과 저장: {summary_path}")

    log.info(f"\n전체 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
