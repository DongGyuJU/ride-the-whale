"""
Stage 5: Elastic Net 학습 + Ridge 비교 + 앙상블
==================================================

실험:
  A. Elastic Net OHLCV
  B. Elastic Net Full (수급 포함)
  C. Ridge Full + Rule 앙상블
  D. Ridge Full + EN Full 앙상블

실행:
  python3 scripts/06_train_elasticnet.py [--market KOSPI|KOSDAQ] [--horizon 5|10|20|ALL]
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
from models.elastic_net import train_elasticnet, predict_elasticnet, get_feature_importance
from evaluate.ic import compute_daily_ic, summarize_ic
from evaluate.long_short import compute_long_short_returns, summarize_long_short

RAW_DIR     = ROOT / "data" / "raw"
CACHE_DIR   = ROOT / "cache" / "features"
RESULTS_DIR = ROOT / "results" / "elasticnet"
HORIZONS    = [5, 10, 20]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "06_train_elasticnet.log", mode="a"),
        ],
    )


def load_universe(market: str) -> list[str] | None:
    path = ROOT / "results" / "diagnose" / f"universe_filter_{market.lower()}.csv"
    if path.exists():
        return pd.read_csv(path)["ticker"].tolist()
    return None


def run_elasticnet(
    market: str,
    mode: str,
    horizon: int,
    panel: pd.DataFrame,
    universe: list[str] | None,
) -> tuple[dict, pd.Series]:
    log = logging.getLogger(__name__)
    exp_name = f"{market}_en_{mode}_fwd{horizon}"
    log.info(f"\n{'='*60}")
    log.info(f"  실험: {exp_name}")
    log.info(f"{'='*60}")

    # 피처 (캐시 재사용)
    cache_path = str(CACHE_DIR / f"{market.lower()}_{mode}.parquet")
    feat_df, feat_cols = build_features(
        panel, mode=mode, universe=universe,
        zscore=True, cache_path=cache_path,
    )

    # 레이블
    tickers = feat_df.index.get_level_values("ticker").unique()
    panel_uni = panel[panel.index.get_level_values("ticker").isin(tickers)]
    labels = make_forward_returns(panel_uni, horizons=[horizon])
    labels = winsorize_returns(labels)
    y_all = labels[f"fwd{horizon}"]

    # 정렬 + NaN 제거
    common = feat_df.index.intersection(y_all.index)
    X_all  = feat_df.loc[common, feat_cols]
    y_all  = y_all.loc[common]
    valid  = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[valid], y_all[valid]

    log.info(f"  유효 데이터: {len(X_all):,} rows, {len(feat_cols)} 피처")

    # 분할
    splits = temporal_split(X_all, y_all.to_frame())
    X_tr, y_tr_df = splits["train"]
    X_vl, y_vl_df = splits["val"]
    X_te, y_te_df = splits["test"]
    y_tr = y_tr_df.iloc[:, 0]
    y_vl = y_vl_df.iloc[:, 0]
    y_te = y_te_df.iloc[:, 0]

    # 학습
    log.info("\n  Elastic Net 학습 중 (grid search)...")
    model, scaler, best_params, val_ic = train_elasticnet(
        X_tr, y_tr, X_vl, y_vl, feat_cols
    )

    # 테스트
    pred_te = predict_elasticnet(model, scaler, X_te, feat_cols)
    daily_ic = compute_daily_ic(pred_te, y_te)
    ic_stats = summarize_ic(daily_ic, label=exp_name)
    ls_df = compute_long_short_returns(pred_te, y_te, q=0.1)
    ls_stats = summarize_long_short(ls_df, label=exp_name)

    # 피처 중요도
    imp = get_feature_importance(model, feat_cols)
    alive = imp[imp > 0]
    zero  = imp[imp == 0]

    log.info(f"\n  살아남은 피처: {len(alive)}/{len(feat_cols)}")
    log.info(f"  L1 제거 피처: {len(zero)}")
    log.info(f"\n  Top-10 (|coef|):")
    for fname, val in imp.head(10).items():
        tag = " ◀ 수급" if fname.startswith("sup_") else ""
        log.info(f"    {fname:<30} {val:.5f}{tag}")

    if mode == "full":
        sup_alive = alive[alive.index.str.startswith("sup_")]
        log.info(f"\n  살아남은 수급 피처: {len(sup_alive)}")
        for fname, val in sup_alive.items():
            log.info(f"    {fname:<30} {val:.5f}")

    # 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    daily_ic.to_csv(RESULTS_DIR / f"daily_ic_{exp_name}.csv")
    ls_df.to_csv(RESULTS_DIR / f"ls_{exp_name}.csv")
    imp.to_csv(RESULTS_DIR / f"importance_{exp_name}.csv")

    result = {
        "experiment": exp_name,
        "market": market, "mode": mode, "horizon": horizon,
        "best_alpha": best_params.get("alpha"),
        "best_l1_ratio": best_params.get("l1_ratio"),
        "n_features_alive": len(alive),
        "val_ic": val_ic,
        **ic_stats,
        **{f"ls_{k}": v for k, v in ls_stats.items()},
    }
    return result, pred_te


def run_ensemble(
    pred_ridge: pd.Series,
    pred_en: pd.Series,
    pred_rule: pd.Series | None,
    y_te: pd.Series,
    market: str,
    horizon: int,
) -> dict:
    """Ridge + EN (+ Rule) 앙상블."""
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*60}")
    log.info(f"  앙상블: {market} fwd{horizon}")
    log.info(f"{'='*60}")

    results = {}

    # Ridge + EN 앙상블 (동일 가중)
    common = pred_ridge.index.intersection(pred_en.index)
    if len(common) > 100:
        ens = (pred_ridge.loc[common] + pred_en.loc[common]) / 2
        y_c = y_te.loc[common.intersection(y_te.index)]
        ens = ens.loc[y_c.index]

        daily_ic = compute_daily_ic(ens, y_c)
        ic_stats = summarize_ic(daily_ic, label=f"{market}_ens_ridge_en_fwd{horizon}")
        ls_df = compute_long_short_returns(ens, y_c, q=0.1)
        ls_stats = summarize_long_short(ls_df, label=f"ensemble_fwd{horizon}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        daily_ic.to_csv(RESULTS_DIR / f"daily_ic_{market}_ens_fwd{horizon}.csv")

        results["ridge_en"] = {
            "experiment": f"{market}_ens_ridge_en_fwd{horizon}",
            "type": "ensemble_ridge_en",
            "market": market, "horizon": horizon,
            **ic_stats,
            **{f"ls_{k}": v for k, v in ls_stats.items()},
        }

    # Ridge + EN + Rule 3-way 앙상블
    if pred_rule is not None:
        common3 = pred_ridge.index.intersection(pred_en.index).intersection(pred_rule.index)
        if len(common3) > 100:
            ens3 = (
                pred_ridge.loc[common3] +
                pred_en.loc[common3] +
                pred_rule.loc[common3]
            ) / 3
            y_c3 = y_te.loc[common3.intersection(y_te.index)]
            ens3 = ens3.loc[y_c3.index]

            daily_ic3 = compute_daily_ic(ens3, y_c3)
            ic_stats3 = summarize_ic(daily_ic3, label=f"{market}_ens3_fwd{horizon}")
            ls_df3 = compute_long_short_returns(ens3, y_c3, q=0.1)
            ls_stats3 = summarize_long_short(ls_df3, label=f"ensemble3_fwd{horizon}")

            results["ridge_en_rule"] = {
                "experiment": f"{market}_ens3_fwd{horizon}",
                "type": "ensemble_ridge_en_rule",
                "market": market, "horizon": horizon,
                **ic_stats3,
                **{f"ls_{k}": v for k, v in ls_stats3.items()},
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ", "ALL"], default="KOSPI")
    parser.add_argument("--horizon", choices=["5", "10", "20", "ALL"], default="ALL")
    args = parser.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    setup_logging()
    log = logging.getLogger(__name__)

    markets  = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]
    horizons = HORIZONS if args.horizon == "ALL" else [int(args.horizon)]

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

        for horizon in horizons:
            en_preds = {}

            for mode in ["ohlcv", "full"]:
                try:
                    result, pred_te = run_elasticnet(
                        market, mode, horizon, panel, universe
                    )
                    all_results.append(result)
                    en_preds[mode] = pred_te
                except Exception as e:
                    log.exception(f"  ❌ EN {mode} fwd{horizon}: {e}")

            # 앙상블 (Ridge 예측값 로드)
            ridge_pred_path = (
                ROOT / "results" / "ridge" /
                f"daily_ic_{market}_full_fwd{horizon}.csv"
            )
            # Ridge daily_ic만 저장했으므로 예측값 재생성 필요 없음
            # → EN full 단독 + Rule과 앙상블만

            if "full" in en_preds:
                # Rule 신호 로드
                rule_ic_path = (
                    ROOT / "results" / "rule" /
                    f"daily_ic_{market}_rule_fwd{horizon}.csv"
                )
                pred_rule = None
                if rule_ic_path.exists():
                    log.info(f"  룰 신호 로드: {rule_ic_path}")

                # Ridge IC와 EN IC 비교
                ridge_summary = ROOT / "results" / "ridge" / "summary.csv"
                if ridge_summary.exists():
                    rdf = pd.read_csv(ridge_summary)
                    r_row = rdf[
                        (rdf["market"] == market) &
                        (rdf["mode"] == "full") &
                        (rdf["horizon"] == horizon)
                    ]
                    en_ic = all_results[-1].get("IC_mean", 0) if all_results else 0
                    if not r_row.empty:
                        r_ic = r_row["IC_mean"].values[0]
                        winner = "EN ✅" if en_ic > r_ic else "Ridge ✅"
                        log.info(
                            f"\n  [비교] Ridge Full {r_ic:.4f} vs EN Full {en_ic:.4f}"
                            f"  → {winner}"
                        )

    # 최종 요약
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = RESULTS_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        log.info(f"\n{'='*60}")
        log.info("  Elastic Net 최종 결과")
        log.info(f"{'='*60}")
        cols = ["experiment", "IC_mean", "IC_IR", "IC_t",
                "best_l1_ratio", "n_features_alive", "ls_sharpe"]
        avail = [c for c in cols if c in summary_df.columns]
        log.info("\n" + summary_df[avail].to_string(index=False))
        log.info(f"\n  결과 저장: {summary_path}")

    log.info(f"\n전체 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
