"""
포지션 사이징
==============

입력: 신호 CSV (07_daily_signal.py 출력)
출력: 종목별 투자 금액 + 주수

방법:
  1. 동일가중 (equal weight) — 기본
  2. 신호 강도 비례 (score weight) — 강한 신호에 더 넣기
  3. 변동성 역비례 (vol weight) — 변동성 높은 종목 비중 축소

실행:
  python3 scripts/08_position_size.py
    --signal results/signals/signal_kospi_20241230_fwd20.csv
    --capital 2000000
    --method equal|score|vol
    --direction LONG|SHORT|BOTH
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

RAW_DIR     = ROOT / "data" / "raw"
SIGNALS_DIR = ROOT / "results" / "signals"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 가중치 계산
# ─────────────────────────────────────────────────────────────────────────────
def equal_weight(scores: pd.Series) -> pd.Series:
    """동일 가중."""
    n = len(scores)
    return pd.Series(1.0 / n, index=scores.index)


def score_weight(scores: pd.Series, temperature: float = 1.0) -> pd.Series:
    """
    점수 강도 비례 가중 (softmax 스타일).
    temperature 낮을수록 상위 집중.
    """
    s = scores / (scores.std() + 1e-10)   # 정규화
    exp_s = np.exp(s / temperature)
    return exp_s / exp_s.sum()


def vol_weight(
    scores: pd.Series,
    market: str,
    lookback: int = 20,
) -> pd.Series:
    """
    변동성 역비례 가중.
    최근 lookback일 수익률 std 계산 후 역수로 비중.
    """
    log = logging.getLogger(__name__)
    panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
    if not panel_path.exists():
        log.warning("  패널 없음 — equal weight로 fallback")
        return equal_weight(scores)

    panel = pd.read_parquet(panel_path)
    latest_date = panel.index.get_level_values("date").max()
    cutoff = latest_date - pd.Timedelta(days=lookback * 2)
    recent = panel[panel.index.get_level_values("date") >= cutoff]

    vols = {}
    for tkr in scores.index:
        try:
            sub = recent.loc[(slice(None), tkr), "close"]
            vol = sub.pct_change().std()
            vols[tkr] = vol if vol > 0 else np.nan
        except Exception:
            vols[tkr] = np.nan

    vol_s = pd.Series(vols).reindex(scores.index).fillna(scores.mean())
    inv_vol = 1.0 / (vol_s + 1e-10)
    return inv_vol / inv_vol.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 주수 계산
# ─────────────────────────────────────────────────────────────────────────────
def calc_shares(
    tickers: list[str],
    weights: pd.Series,
    capital: float,
    market: str,
    min_order: int = 1,
) -> pd.DataFrame:
    """
    종목별 투자 금액 + 주수 계산.

    Args:
        tickers: 매수 종목 리스트
        weights: 종목별 비중 (합=1)
        capital: 총 투자 금액 (원)
        market: KOSPI | KOSDAQ
        min_order: 최소 주문 단위

    Returns:
        DataFrame[ticker, weight, alloc_krw, price, shares, actual_krw]
    """
    log = logging.getLogger(__name__)

    # 최신 종가 조회
    panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
    prices = {}
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        latest = panel.index.get_level_values("date").max()
        for tkr in tickers:
            try:
                price = panel.loc[(latest, tkr), "close"]
                prices[tkr] = float(price)
            except Exception:
                prices[tkr] = np.nan

    rows = []
    for tkr in tickers:
        w      = weights.get(tkr, 0)
        alloc  = capital * w
        price  = prices.get(tkr, np.nan)
        if pd.isna(price) or price <= 0:
            shares = 0
            actual = 0.0
        else:
            shares = max(min_order, int(alloc / price))
            actual = shares * price

        rows.append({
            "ticker":     tkr,
            "weight":     round(w, 4),
            "alloc_krw":  round(alloc),
            "price":      round(price) if not pd.isna(price) else 0,
            "shares":     shares,
            "actual_krw": round(actual),
        })

    result = pd.DataFrame(rows)
    total_actual = result["actual_krw"].sum()
    log.info(f"  총 투자 금액: {capital:,.0f}원")
    log.info(f"  실제 집행 금액: {total_actual:,.0f}원 ({total_actual/capital:.1%})")
    log.info(f"  미집행 (현금): {capital-total_actual:,.0f}원")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def compute_positions(
    signal_df: pd.DataFrame,
    capital: float,
    method: str,
    direction: str,
    market: str,
) -> pd.DataFrame:
    log = logging.getLogger(__name__)

    # 방향 필터
    if direction == "LONG":
        df = signal_df[signal_df["direction"] == "LONG"].copy()
    elif direction == "SHORT":
        df = signal_df[signal_df["direction"] == "SHORT"].copy()
        df["score"] = -df["score"]   # short은 점수 반전
    else:
        df = signal_df[signal_df["direction"] != "neutral"].copy()

    if df.empty:
        log.error("  신호 없음.")
        return pd.DataFrame()

    scores = df.set_index("ticker")["score"]

    # 가중치
    if method == "equal":
        weights = equal_weight(scores)
    elif method == "score":
        weights = score_weight(scores, temperature=0.5)
    elif method == "vol":
        weights = vol_weight(scores, market=market)
    else:
        weights = equal_weight(scores)

    log.info(f"\n  방법: {method} | 방향: {direction} | 종목 수: {len(scores)}")

    # 주수 계산
    pos_df = calc_shares(
        tickers=scores.index.tolist(),
        weights=weights,
        capital=capital,
        market=market,
    )

    # 종목명 붙이기
    if "name" in df.columns:
        name_map = df.set_index("ticker")["name"].to_dict()
        pos_df["name"] = pos_df["ticker"].map(name_map).fillna("")

    pos_df["direction"] = direction
    pos_df["score"] = pos_df["ticker"].map(scores.to_dict())
    pos_df = pos_df.sort_values("weight", ascending=False).reset_index(drop=True)
    return pos_df


def print_positions(pos_df: pd.DataFrame, capital: float) -> None:
    log = logging.getLogger(__name__)

    log.info(f"\n{'='*65}")
    log.info(f"  포지션 상세  (총 자본: {capital:,.0f}원)")
    log.info(f"{'='*65}")
    log.info(f"  {'티커':<8} {'종목명':<18} {'비중':>6} {'배분금액':>12} {'종가':>8} {'주수':>6} {'실제금액':>12}")
    log.info(f"  {'-'*65}")

    for _, row in pos_df.iterrows():
        name = str(row.get("name", ""))[:16]
        log.info(
            f"  {row['ticker']:<8} {name:<18} "
            f"{row['weight']:>5.1%} "
            f"{row['alloc_krw']:>12,.0f} "
            f"{row['price']:>8,.0f} "
            f"{row['shares']:>6} "
            f"{row['actual_krw']:>12,.0f}"
        )

    total = pos_df["actual_krw"].sum()
    log.info(f"  {'합계':<28} {total:>12,.0f}원")
    remaining = capital - total
    log.info(f"  {'현금 잔여':<28} {remaining:>12,.0f}원 ({remaining/capital:.1%})")


def main():
    parser = argparse.ArgumentParser(description="포지션 사이징")
    parser.add_argument("--signal",    type=str, default=None,
                        help="신호 CSV 경로 (없으면 최신 파일 자동 선택)")
    parser.add_argument("--capital",   type=float, default=2_000_000,
                        help="투자 가능 금액 (원, 기본: 200만)")
    parser.add_argument("--method",    choices=["equal", "score", "vol"], default="equal")
    parser.add_argument("--direction", choices=["LONG", "SHORT", "BOTH"], default="LONG")
    parser.add_argument("--market",    choices=["KOSPI", "KOSDAQ"],       default="KOSPI")
    parser.add_argument("--save",      action="store_true", help="결과 CSV 저장")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    # 신호 파일 선택
    if args.signal:
        signal_path = Path(args.signal)
    else:
        # 최신 파일 자동 선택
        candidates = sorted(SIGNALS_DIR.glob(f"signal_{args.market.lower()}_*.csv"))
        if not candidates:
            log.error(f"  ❌ 신호 파일 없음. 먼저 07_daily_signal.py 실행하세요.")
            sys.exit(1)
        signal_path = candidates[-1]
        log.info(f"  최신 신호 파일 자동 선택: {signal_path.name}")

    signal_df = pd.read_csv(signal_path)
    log.info(f"  신호 날짜: {signal_df['signal_date'].iloc[0]}")
    log.info(f"  전체 종목: {len(signal_df)}")

    # 포지션 계산
    pos_df = compute_positions(
        signal_df, args.capital, args.method,
        args.direction, args.market,
    )

    if pos_df.empty:
        sys.exit(1)

    print_positions(pos_df, args.capital)

    # 저장
    if args.save:
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = signal_df["signal_date"].iloc[0]
        save_path = SIGNALS_DIR / f"position_{args.market.lower()}_{date_str}_{args.method}.csv"
        pos_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        log.info(f"\n  💾 저장: {save_path}")


if __name__ == "__main__":
    main()
