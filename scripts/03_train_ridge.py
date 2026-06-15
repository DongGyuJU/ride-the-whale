"""
Stage 2: Ridge Baseline 학습 + Ablation
=========================================

실험:
  A. KOSPI  OHLCV only
  B. KOSPI  OHLCV + 수급
  C. KOSDAQ OHLCV only
  D. KOSDAQ OHLCV + 수급

각 실험마다:
  - fwd5 / fwd10 / fwd20 각각 학습
  - test IC, IC IR, Long-Short Sharpe 출력
  - 결과 results/ridge/ 에 저장

실행:
  python3 scripts/03_train_ridge.py [--market KOSPI|KOSDAQ|ALL] [--horizon 5|10|20|ALL]
"""
from __future__ import annotations

import argparse
import json
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
from models.temporal_split import temporal_split
from models.ridge import train_ridge, predict_ridge
from evaluate.ic import compute_daily_ic, summarize_ic
from evaluate.long_short import compute_long_short_returns, summarize_long_short

RAW_DIR     = ROOT / "data" / "raw"
CACHE_DIR   = ROOT / "cache" / "features"
RESULTS_DIR = ROOT / "results" / "ridge"

HORIZONS = [5, 10, 20]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "03_train_ridge.log", mode="a"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(
    market: str,
    mode: str,        # 'ohlcv' | 'full'
    horizon: int,
    panel: pd.DataFrame,
    universe: list[str],
) -> dict:
    """단일 실험 실행."""
    log = logging.getLogger(__name__)
    exp_name = f"{market}_{mode}_fwd{horizon}"
    log.info(f"\n{'='*60}")
    log.info(f"  실험: {exp_name}")
    log.info(f"{'='*60}")

    # 1. 피처 빌드
    cache_path = str(CACHE_DIR / f"{market.lower()}_{mode}.parquet")
    feat_df, feat_cols = build_features(
        panel,
        mode=mode,
        universe=universe,
        zscore=True,
        cache_path=cache_path if horizon == HORIZONS[0] else None,
        # 첫 horizon에서만 캐시 저장 (피처는 horizon과 무관)
    )
    if not feat_cols:
        log.error(f"  피처 없음 — 스킵")
        return {}

    # 2. 레이블
    # panel에서 close만 추출 (유니버스 필터)
    tickers = feat_df.index.get_level_values("ticker").unique()
    panel_uni = panel[panel.index.get_level_values("ticker").isin(tickers)]
    labels = make_forward_returns(panel_uni, horizons=[horizon])
    labels = winsorize_returns(labels)
    y_all = labels[f"fwd{horizon}"]

    # 3. 피처-레이블 공통 인덱스 정렬
    common = feat_df.index.intersection(y_all.index)
    X_all = feat_df.loc[common, feat_cols]
    y_all = y_all.loc[common]

    # NaN 제거
    valid = X_all.notna().all(axis=1) & y_all.notna()
    X_all = X_all[valid]
    y_all = y_all[valid]

    log.info(f"  유효 데이터: {len(X_all):,} rows, {len(feat_cols)} 피처")

    # 4. 시간 분할
    splits = temporal_split(X_all, y_all.to_frame())
    X_tr, y_tr_df = splits["train"]
    X_vl, y_vl_df = splits["val"]
    X_te, y_te_df = splits["test"]

    y_tr = y_tr_df.iloc[:, 0]
    y_vl = y_vl_df.iloc[:, 0]
    y_te = y_te_df.iloc[:, 0]

    # 5. 학습
    log.info(f"\n  Ridge 학습 중 (alpha 탐색)...")
    model, scaler, best_alpha, val_ic = train_ridge(
        X_tr, y_tr, X_vl, y_vl, feat_cols
    )

    # 6. Test 예측
    pred_te = predict_ridge(model, scaler, X_te, feat_cols)

    # 7. IC 평가
    daily_ic = compute_daily_ic(pred_te, y_te)
    ic_stats = summarize_ic(daily_ic, label=exp_name)

    # 8. Long-Short
    ls_df = compute_long_short_returns(pred_te, y_te, q=0.1)
    ls_stats = summarize_long_short(ls_df, label=exp_name)

    # 9. 피처 중요도 (Ridge coefficients)
    coef = pd.Series(
        np.abs(model.coef_), index=feat_cols
    ).sort_values(ascending=False)
    log.info(f"\n  Top-10 피처 (|coef|):")
    for fname, val in coef.head(10).items():
        log.info(f"    {fname:<25} {val:.4f}")

    # 10. 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": exp_name,
        "market": market,
        "mode": mode,
        "horizon": horizon,
        "n_features": len(feat_cols),
        "best_alpha": best_alpha,
        "val_ic": val_ic,
        **ic_stats,
        **{f"ls_{k}": v for k, v in ls_stats.items()},
    }

    # daily IC 저장
    daily_ic.to_csv(RESULTS_DIR / f"daily_ic_{exp_name}.csv")
    ls_df.to_csv(RESULTS_DIR / f"ls_{exp_name}.csv")
    coef.to_csv(RESULTS_DIR / f"coef_{exp_name}.csv")

    return result


# ─────────────────────────────────────────────────────────────────────────────
def load_universe(market: str) -> list[str]:
    """진단 결과에서 유니버스 로드."""
    path = ROOT / "results" / "diagnose" / f"universe_filter_{market.lower()}.csv"
    if path.exists():
        return pd.read_csv(path)["ticker"].tolist()
    return None  # None이면 전체 사용


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge baseline 학습")
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
        # 패널 로드
        panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
        if not panel_path.exists():
            log.error(f"  ❌ {market} 패널 없음. 01_download.py 먼저 실행하세요.")
            continue

        log.info(f"\n{'#'*60}")
        log.info(f"  {market} 패널 로드 중...")
        panel = pd.read_parquet(panel_path)
        log.info(f"  패널: {panel.shape}")

        universe = load_universe(market)
        if universe:
            log.info(f"  유니버스: {len(universe)} 종목 (diagnose 결과)")
        else:
            log.info(f"  유니버스: 전체 종목 (diagnose 결과 없음)")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            for horizon in horizons:
                try:
                    result = run_experiment(market, mode, horizon, panel, universe)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    log.exception(f"  ❌ {market}/{mode}/fwd{horizon} 실패: {e}")

    # ── 최종 비교표
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = RESULTS_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        log.info(f"\n{'='*60}")
        log.info("  최종 결과 비교")
        log.info(f"{'='*60}")

        cols = ["experiment", "IC_mean", "IC_IR", "IC_t", "ls_sharpe"]
        avail = [c for c in cols if c in summary_df.columns]
        log.info("\n" + summary_df[avail].to_string(index=False))
        log.info(f"\n  결과 저장: {summary_path}")

        # OHLCV vs Full ablation 하이라이트
        for market in markets:
            for h in horizons:
                ohlcv_row = summary_df[
                    (summary_df["market"] == market) &
                    (summary_df["mode"] == "ohlcv") &
                    (summary_df["horizon"] == h)
                ]
                full_row = summary_df[
                    (summary_df["market"] == market) &
                    (summary_df["mode"] == "full") &
                    (summary_df["horizon"] == h)
                ]
                if ohlcv_row.empty or full_row.empty:
                    continue
                ic_base = ohlcv_row["IC_mean"].values[0]
                ic_full = full_row["IC_mean"].values[0]
                improvement = ic_full - ic_base
                log.info(f"\n  [{market} fwd{h}] 수급 추가 효과: "
                         f"OHLCV {ic_base:.4f} → Full {ic_full:.4f} "
                         f"(Δ{improvement:+.4f})")

    log.info(f"\n전체 소요: {(time.time()-t0)/60:.1f}분")
    log.info("다음 단계: python3 scripts/04_train_lgb.py")


if __name__ == "__main__":
    main()
