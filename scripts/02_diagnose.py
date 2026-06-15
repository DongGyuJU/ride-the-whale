"""
Stage 1-B: 데이터 품질 진단
=====================================
01_download.py 실행 후 반드시 이 스크립트 실행.

출력:
  results/diagnose/summary.txt     ── 텍스트 요약 (콘솔에도 출력)
  results/diagnose/coverage.csv    ── 종목별 수급 coverage 통계
  results/diagnose/ticker_stats.csv── 종목별 기본 통계
  results/diagnose/universe_filter.csv ── 유니버스 필터 후보 목록

실행:
  python3 scripts/02_diagnose.py [--market KOSPI|KOSDAQ|ALL]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from io import StringIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

RAW_DIR     = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "diagnose"

SUPPLY_COLS = ["foreign_net", "inst_net", "retail_net", "other_net"]
OHLCV_COLS  = ["open", "high", "low", "close", "volume", "trade_value"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(n: float, pct: bool = False) -> str:
    if pct:
        return f"{n:.1%}"
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}조"
    if abs(n) >= 1e8:
        return f"{n/1e8:.1f}억"
    return f"{n:,.0f}"


def _pct_nonzero(s: pd.Series) -> float:
    """NaN 제외하고 non-zero 비율."""
    valid = s.dropna()
    if len(valid) == 0:
        return 0.0
    return (valid != 0).mean()


def _pct_nonnull(s: pd.Series) -> float:
    return s.notna().mean()


# ─────────────────────────────────────────────────────────────────────────────
# 진단 로직
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_market(market: str, buf: StringIO) -> pd.DataFrame | None:
    log = logging.getLogger(__name__)

    panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
    meta_path  = RAW_DIR / f"meta_{market.lower()}.parquet"

    if not panel_path.exists():
        log.warning(f"  ❌ {market} 패널 없음: {panel_path}")
        log.warning(f"     01_download.py 먼저 실행하세요.")
        return None

    log.info(f"\n{'='*60}")
    log.info(f"  {market} 진단 시작")
    log.info(f"{'='*60}")

    panel = pd.read_parquet(panel_path)
    meta  = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()

    def p(line: str = "") -> None:
        print(line, file=buf)
        log.info(line)

    # ── 1. 기본 구조
    p(f"\n{'━'*55}")
    p(f"  [{market}] 기본 구조")
    p(f"{'━'*55}")

    dates   = panel.index.get_level_values("date").unique().sort_values()
    tickers = panel.index.get_level_values("ticker").unique()

    p(f"  날짜 범위  : {dates[0].date()} ~ {dates[-1].date()}")
    p(f"  총 거래일  : {len(dates)}")
    p(f"  총 종목 수 : {len(tickers)}")
    p(f"  총 레코드  : {len(panel):,}")
    p(f"  컬럼       : {list(panel.columns)}")
    if not meta.empty and "market_cap" in meta.columns:
        mcap = meta["market_cap"]
        p(f"  시총 범위  : {_fmt(mcap.min())} ~ {_fmt(mcap.max())}")

    # ── 2. NaN 현황
    p(f"\n  [NaN 현황]")
    for col in OHLCV_COLS + SUPPLY_COLS:
        if col not in panel.columns:
            p(f"    {col:<15} : 컬럼 없음")
            continue
        nan_rate = panel[col].isna().mean()
        zero_rate = (panel[col].fillna(0) == 0).mean()
        p(f"    {col:<15} : NaN {nan_rate:.1%}  |  zero {zero_rate:.1%}")

    # ── 3. 수급 coverage (핵심)
    p(f"\n  [수급 Coverage — 핵심 지표]")

    # 종목별 수급 coverage
    ticker_coverage = []
    for tkr in tickers:
        try:
            sub = panel.loc[(slice(None), tkr), :]
        except KeyError:
            continue

        row = {"ticker": tkr, "n_days": len(sub)}
        for col in SUPPLY_COLS:
            if col in sub.columns:
                row[f"{col}_nonnull"] = _pct_nonnull(sub[col])
                row[f"{col}_nonzero"] = _pct_nonzero(sub[col])
            else:
                row[f"{col}_nonnull"] = 0.0
                row[f"{col}_nonzero"] = 0.0

        # 거래대금
        if "trade_value" in sub.columns:
            row["avg_trade_value"] = sub["trade_value"].median()
        else:
            row["avg_trade_value"] = 0.0

        # 시총
        if "market_cap" in sub.columns:
            row["market_cap"] = sub["market_cap"].iloc[0] if len(sub) > 0 else 0
        elif not meta.empty and "market_cap" in meta.columns:
            mcap_map = meta.set_index("ticker")["market_cap"].to_dict()
            row["market_cap"] = mcap_map.get(tkr, 0)
        else:
            row["market_cap"] = 0

        ticker_coverage.append(row)

    cov_df = pd.DataFrame(ticker_coverage)

    for col in SUPPLY_COLS:
        nonnull_col = f"{col}_nonnull"
        nonzero_col = f"{col}_nonzero"
        if nonnull_col in cov_df.columns:
            # 종목 기준 비율
            pct_tickers_with_data = (cov_df[nonnull_col] > 0.5).mean()
            pct_nonzero_mean      = cov_df[nonzero_col].mean()
            p(f"    {col:<15} : 종목 50%+ 데이터 {pct_tickers_with_data:.1%}  "
              f"|  평균 non-zero {pct_nonzero_mean:.1%}")

    # ── 4. 유동성 분포
    p(f"\n  [유동성 분포 (일평균 거래대금 기준)]")
    if "avg_trade_value" in cov_df.columns:
        tv = cov_df["avg_trade_value"].replace(0, np.nan).dropna()
        pcts = [10, 25, 50, 75, 90]
        quantiles = np.nanpercentile(tv, pcts)
        for pct_val, q in zip(pcts, quantiles):
            p(f"    P{pct_val:02d}: {_fmt(q)}")

        # 필터 후보
        threshold_1b = (tv >= 1e9).mean()
        threshold_500m = (tv >= 5e8).mean()
        p(f"\n    일 거래대금 10억↑ 종목 비율: {threshold_1b:.1%}")
        p(f"    일 거래대금 5억↑  종목 비율: {threshold_500m:.1%}")

    # ── 5. 이상치
    p(f"\n  [이상치 탐지]")
    if "close" in panel.columns:
        # 일봉 수익률 극단값
        pct_ret = panel["close"].groupby(level="ticker").pct_change()
        extreme = (pct_ret.abs() > 0.30)
        p(f"    일수익률 ±30% 초과 레코드: {extreme.sum():,} ({extreme.mean():.3%})")
        p(f"    일수익률 ±50% 초과 레코드: {(pct_ret.abs() > 0.50).sum():,}")

    if "foreign_net" in panel.columns:
        fnet = panel["foreign_net"].dropna()
        q99 = fnet.abs().quantile(0.99)
        p(f"    외인 순매수 P99 (절대값): {_fmt(q99)}")

    # ── 6. 유니버스 필터 권고
    p(f"\n  [유니버스 필터 권고]")
    if "avg_trade_value" in cov_df.columns and "foreign_net_nonnull" in cov_df.columns:
        # 조건: 거래대금 5억↑ AND 외인 수급 데이터 존재
        mask = (
            (cov_df["avg_trade_value"] >= 5e8) &
            (cov_df["foreign_net_nonnull"] > 0.5)
        )
        universe = cov_df[mask]
        p(f"    필터 후 (거래대금≥5억 & 외인수급↑): {len(universe)}/{len(cov_df)} 종목")
        p(f"    → 최종 유니버스 권고: {len(universe)} 종목 ({len(universe)/len(cov_df):.1%})")

        # 저장
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        universe_path = RESULTS_DIR / f"universe_filter_{market.lower()}.csv"
        cov_df.to_csv(RESULTS_DIR / f"coverage_{market.lower()}.csv", index=False)
        universe = universe.copy()
        universe["ticker"] = universe["ticker"].astype(str).str.zfill(6)
        universe[["ticker"]].to_csv(universe_path, index=False)
        p(f"\n    universe 저장: {universe_path}")

    return cov_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KRX 데이터 품질 진단")
    parser.add_argument(
        "--market",
        choices=["KOSPI", "KOSDAQ", "ALL"],
        default="ALL",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    log = logging.getLogger(__name__)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    markets = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]

    buf = StringIO()

    all_coverage = {}
    for market in markets:
        cov = diagnose_market(market, buf)
        if cov is not None:
            all_coverage[market] = cov

    # ── 비교 요약 (ALL일 때)
    if len(all_coverage) == 2:
        print("\n" + "="*55, file=buf)
        print("  KOSPI vs KOSDAQ 비교", file=buf)
        print("="*55, file=buf)
        for market, cov in all_coverage.items():
            n = len(cov)
            has_supply = (cov.get("foreign_net_nonnull", pd.Series([0])) > 0.5).mean() if "foreign_net_nonnull" in cov.columns else 0
            print(f"  {market:<8}: {n} 종목  |  외인수급 50%+ {has_supply:.1%}", file=buf)

    # 텍스트 요약 저장
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    log.info(f"\n📄 요약 저장: {summary_path}")
    log.info("다음 단계: Stage 2 피처 엔지니어링 준비 완료!")


if __name__ == "__main__":
    main()
