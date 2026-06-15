"""
Stage 4: 룰 기반 백테스트
============================

룰:
  1. 외인 5일 연속 순매수 (sup_fstreak >= 5)
  2. 종가가 60일 이동평균 위 (tech_ma60_dev > 0)
  3. 두 조건 동시 만족 → 매수 시그널

목적:
  - ML 알파와 독립적인 신호인지 확인
  - Ridge 예측값과 상관관계 측정 (낮을수록 앙상블 효과 큼)
  - 단독 성과 측정 (IC, L/S)

실행:
  python3 scripts/05_rule_backtest.py [--market KOSPI|KOSDAQ]
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

from features.technical import add_technical_features
from features.supply import add_supply_features
from labels.forward_return import make_forward_returns, winsorize_returns
from models.temporal_split import SPLIT
from evaluate.ic import compute_daily_ic, summarize_ic
from evaluate.long_short import compute_long_short_returns, summarize_long_short

RAW_DIR     = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "rule"
HORIZONS    = [5, 10, 20]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "05_rule_backtest.log", mode="a"),
        ],
    )


def load_universe(market: str) -> list[str] | None:
    path = ROOT / "results" / "diagnose" / f"universe_filter_{market.lower()}.csv"
    if path.exists():
        return pd.read_csv(path)["ticker"].tolist()
    return None


def build_rule_signal(panel: pd.DataFrame) -> pd.Series:
    """
    룰 기반 시그널 점수 (연속적 버전 — 랭킹 가능하게).

    점수 구성:
      - sup_fstreak: 외인 연속 매수일 수 (음수면 연속 매도)
      - tech_ma60_dev: 60일선 대비 괴리율 (양수 = 위)
      - sup_fcum20: 20일 누적 외인 순매수 비율

    → 세 점수의 CS z-score 합산
    """
    log = logging.getLogger(__name__)
    log.info("[rule] 기술적 + 수급 피처 계산 중...")

    df = add_technical_features(panel)
    df = add_supply_features(df)

    # 각 컴포넌트 CS z-score
    date_grp = df.groupby(level="date", group_keys=False)

    components = []

    # 1. 외인 연속 매수 streak
    if "sup_fstreak" in df.columns:
        z1 = date_grp["sup_fstreak"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-10)
        ).clip(-3, 3)
        components.append(z1)
        log.info("  ✅ 컴포넌트 1: sup_fstreak")

    # 2. 60일 이동평균 위
    if "tech_ma60_dev" in df.columns:
        z2 = date_grp["tech_ma60_dev"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-10)
        ).clip(-3, 3)
        components.append(z2)
        log.info("  ✅ 컴포넌트 2: tech_ma60_dev")

    # 3. 20일 누적 외인 순매수
    if "sup_fcum20" in df.columns:
        z3 = date_grp["sup_fcum20"].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-10)
        ).clip(-3, 3)
        components.append(z3)
        log.info("  ✅ 컴포넌트 3: sup_fcum20")

    if not components:
        raise RuntimeError("필요한 피처가 없습니다.")

    # 합산
    signal = sum(components) / len(components)
    signal.name = "rule_signal"
    log.info(f"  신호 생성 완료: {signal.notna().sum():,} valid rows")
    return signal


def compute_ridge_correlation(
    rule_signal: pd.Series,
    market: str,
    horizon: int,
) -> float | None:
    """Ridge 예측값과 룰 신호의 상관계수."""
    ridge_path = ROOT / "results" / "ridge" / f"daily_ic_KOSPI_full_fwd{horizon}.csv"
    # daily_ic가 아니라 pred가 필요 — 없으면 스킵
    return None


def run_backtest(
    market: str,
    panel: pd.DataFrame,
    universe: list[str] | None,
) -> list[dict]:
    log = logging.getLogger(__name__)
    log.info(f"\n{'='*60}")
    log.info(f"  {market} 룰 백테스트")
    log.info(f"{'='*60}")

    # 유니버스 필터
    if universe:
        tickers = panel.index.get_level_values("ticker")
        panel = panel[tickers.isin(universe)]
        log.info(f"  유니버스: {len(universe)} 종목")

    # 신호 생성
    signal = build_rule_signal(panel)

    # 레이블
    labels = make_forward_returns(panel, horizons=HORIZONS)
    labels = winsorize_returns(labels)

    # 테스트 기간만 평가
    test_start, test_end = SPLIT["test"]
    dates = signal.index.get_level_values("date")
    test_mask = (dates >= test_start) & (dates <= test_end)
    signal_test = signal[test_mask]

    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        y = labels[f"fwd{h}"]
        y_test = y[test_mask]

        common = signal_test.index.intersection(y_test.index)
        sig = signal_test.loc[common].dropna()
        yy  = y_test.loc[sig.index].dropna()
        common2 = sig.index.intersection(yy.index)
        sig, yy = sig.loc[common2], yy.loc[common2]

        log.info(f"\n  [fwd{h}] 유효 데이터: {len(sig):,} rows")

        # IC
        daily_ic = compute_daily_ic(sig, yy)
        ic_stats = summarize_ic(daily_ic, label=f"{market}_rule_fwd{h}")

        # L/S
        ls_df = compute_long_short_returns(sig, yy, q=0.1)
        ls_stats = summarize_long_short(ls_df, label=f"{market}_rule_fwd{h}")

        # 이진 룰 성과 (streak>=5 AND ma60 위)
        log.info(f"\n  [이진 룰 분석 — fwd{h}]")
        try:
            df_tmp = add_technical_features(panel)
            df_tmp = add_supply_features(df_tmp)
            d = df_tmp.index.get_level_values("date")
            tm = (d >= test_start) & (d <= test_end)

            streak_ok = df_tmp["sup_fstreak"][tm] >= 5 if "sup_fstreak" in df_tmp.columns else None
            ma_ok     = df_tmp["tech_ma60_dev"][tm] > 0 if "tech_ma60_dev" in df_tmp.columns else None

            if streak_ok is not None and ma_ok is not None:
                binary_signal = (streak_ok & ma_ok).astype(float)
                n_signal = binary_signal.sum()
                total = len(binary_signal)
                log.info(f"  신호 발생: {int(n_signal):,}/{total:,} ({n_signal/total:.1%})")

                # 신호 발생 시 평균 수익률
                y_bin = labels[f"fwd{h}"][tm]
                common_b = binary_signal.index.intersection(y_bin.index)
                sig_b = binary_signal.loc[common_b]
                y_b   = y_bin.loc[common_b]

                hit_ret  = y_b[sig_b == 1].mean()
                miss_ret = y_b[sig_b == 0].mean()
                log.info(f"  신호 발생 평균 fwd{h}: {hit_ret:.4f}")
                log.info(f"  신호 미발생 평균 fwd{h}: {miss_ret:.4f}")
                log.info(f"  차이: {hit_ret - miss_ret:+.4f}")
        except Exception as e:
            log.warning(f"  이진 룰 분석 실패: {e}")

        # 저장
        daily_ic.to_csv(RESULTS_DIR / f"daily_ic_{market}_rule_fwd{h}.csv")
        ls_df.to_csv(RESULTS_DIR / f"ls_{market}_rule_fwd{h}.csv")

        results.append({
            "experiment": f"{market}_rule_fwd{h}",
            "market": market,
            "horizon": h,
            **ic_stats,
            **{f"ls_{k}": v for k, v in ls_stats.items()},
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["KOSPI", "KOSDAQ", "ALL"], default="KOSPI")
    args = parser.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    setup_logging()
    log = logging.getLogger(__name__)

    markets = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]
    all_results = []
    t0 = time.time()

    for market in markets:
        panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
        if not panel_path.exists():
            log.error(f"  ❌ {market} 패널 없음.")
            continue

        log.info(f"  {market} 패널 로드 중...")
        panel = pd.read_parquet(panel_path)
        universe = load_universe(market)
        results = run_backtest(market, panel, universe)
        all_results.extend(results)

    # 요약
    if all_results:
        summary_df = pd.DataFrame(all_results)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

        log.info(f"\n{'='*60}")
        log.info("  룰 백테스트 최종 결과")
        log.info(f"{'='*60}")
        cols = ["experiment", "IC_mean", "IC_IR", "IC_t", "ls_sharpe"]
        avail = [c for c in cols if c in summary_df.columns]
        log.info("\n" + summary_df[avail].to_string(index=False))

        # Ridge와 비교
        ridge_path = ROOT / "results" / "ridge" / "summary.csv"
        if ridge_path.exists():
            ridge_df = pd.read_csv(ridge_path)
            log.info(f"\n  [Ridge Full vs 룰 기반 비교]")
            for market in markets:
                for h in HORIZONS:
                    r_row = ridge_df[
                        (ridge_df["market"] == market) &
                        (ridge_df["mode"] == "full") &
                        (ridge_df["horizon"] == h)
                    ]
                    rule_row = summary_df[summary_df["experiment"] == f"{market}_rule_fwd{h}"]
                    if r_row.empty or rule_row.empty:
                        continue
                    r_ic   = r_row["IC_mean"].values[0]
                    ru_ic  = rule_row["IC_mean"].values[0]
                    log.info(f"  fwd{h}: Ridge {r_ic:.4f} | 룰 {ru_ic:.4f}")

    log.info(f"\n전체 소요: {(time.time()-t0)/60:.1f}분")
    log.info("다음 단계: python3 scripts/06_train_elasticnet.py")


if __name__ == "__main__":
    main()
