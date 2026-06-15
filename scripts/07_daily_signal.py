"""
Stage 6: 실전 신호 생성
=========================

매일 실행하면 오늘 날짜 기준 매수/매도 종목 리스트 출력.

동작:
  1. 최근 데이터 다운로드 (오늘 포함 60일치)
  2. 피처 계산 (rolling, look-ahead 없음)
  3. 저장된 Ridge Full 모델로 예측
  4. 상위 N종목 매수 / 하위 N종목 매도 신호 출력
  5. results/signals/ 에 CSV 저장

실행:
  python3 scripts/07_daily_signal.py [--market KOSPI|KOSDAQ|ALL] [--topn 20]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from data.env_loader import load_env
from data.krx_loader import load_market
from features.technical import add_technical_features
from features.supply import add_supply_features
from features.pipeline import cross_sectional_zscore

RESULTS_DIR  = ROOT / "results"
SIGNALS_DIR  = RESULTS_DIR / "signals"
MODEL_DIR    = ROOT / "models" / "saved"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 모델 저장 / 로드
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, scaler, feat_cols: list[str], market: str, mode: str, horizon: int) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"ridge_{market.lower()}_{mode}_fwd{horizon}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feat_cols": feat_cols}, f)
    logging.getLogger(__name__).info(f"  모델 저장: {path}")
    return path


def load_model(market: str, mode: str, horizon: int) -> tuple:
    path = MODEL_DIR / f"ridge_{market.lower()}_{mode}_fwd{horizon}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"저장된 모델 없음: {path}\n"
            f"먼저 scripts/07_save_model.py 실행하세요."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"], obj["feat_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# 최근 데이터 다운로드
# ─────────────────────────────────────────────────────────────────────────────
def fetch_recent(
    market: str,
    lookback_days: int = 90,
    cache_dir: str = None,
) -> pd.DataFrame:
    """최근 lookback_days 거래일치 OHLCV + 수급 다운로드."""
    log = logging.getLogger(__name__)
    end_date   = datetime.today().strftime("%Y%m%d")
    # 영업일 기준 lookback_days ≈ 달력 1.5배
    start_date = (datetime.today() - timedelta(days=int(lookback_days * 1.5))).strftime("%Y%m%d")

    log.info(f"  최근 데이터 다운로드: {start_date} ~ {end_date}")
    cache = str(ROOT / "cache" / "daily") if cache_dir is None else cache_dir

    meta_df, ticker_map = load_market(
        start_date=start_date,
        end_date=end_date,
        market=market,
        cache_dir=cache,
        max_workers=4,
        force_refresh=True,   # 항상 최신 데이터
    )

    # 패널 빌드
    frames = []
    for ticker, df in ticker_map.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        raise RuntimeError("다운로드된 데이터 없음.")

    panel = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
    log.info(f"  패널: {panel.shape} | 날짜: {panel.index.get_level_values('date').unique()[-1].date()}")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 피처 계산 (최신 날짜 하루치만 사용)
# ─────────────────────────────────────────────────────────────────────────────
def compute_latest_features(
    panel: pd.DataFrame,
    feat_cols: list[str],
    universe: list[str] | None = None,
) -> pd.DataFrame:
    """패널에서 기술적 + 수급 피처 계산 후 최신 날짜 행만 반환."""
    log = logging.getLogger(__name__)

    if universe:
        tickers = panel.index.get_level_values("ticker")
        panel = panel[tickers.isin(universe)]

    # 피처 계산
    df = add_technical_features(panel)
    supply_cols = [c for c in ["foreign_net", "inst_net"] if c in df.columns]
    if supply_cols:
        df = add_supply_features(df)

    # 피처 컬럼만 추출
    avail_cols = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(avail_cols)
    if missing:
        log.warning(f"  누락 피처 {len(missing)}개: {missing}")

    feat_df = df[avail_cols].copy()

    # CS z-score (오늘 날짜 기준)
    feat_df = cross_sectional_zscore(feat_df, avail_cols)

    # 최신 날짜만
    latest_date = feat_df.index.get_level_values("date").max()
    latest = feat_df.loc[latest_date]
    log.info(f"  피처 날짜: {latest_date.date()} | 종목 수: {len(latest)}")

    return latest, latest_date


# ─────────────────────────────────────────────────────────────────────────────
# 신호 생성
# ─────────────────────────────────────────────────────────────────────────────
def generate_signal(
    market: str,
    mode: str,
    horizon: int,
    top_n: int,
    panel: pd.DataFrame,
    universe: list[str] | None,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    log = logging.getLogger(__name__)

    # 모델 로드
    model, scaler, feat_cols = load_model(market, mode, horizon)

    # 피처 계산
    latest_feat, signal_date = compute_latest_features(panel, feat_cols, universe)

    # 예측
    avail_cols = [c for c in feat_cols if c in latest_feat.columns]
    X = latest_feat[avail_cols].fillna(0).values
    X_s = scaler.transform(X)
    scores = model.predict(X_s)

    result = pd.DataFrame({
        "ticker": latest_feat.index,
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    # 종목명 붙이기
    if meta is not None and "name" in meta.columns:
        name_map = meta.set_index("ticker")["name"].to_dict()
        result["name"] = result["ticker"].map(name_map).fillna("")

    result["signal_date"] = signal_date.date()
    result["market"] = market
    result["horizon"] = horizon
    result["rank"] = range(1, len(result) + 1)

    # 매수/매도 구분
    result["direction"] = "neutral"
    result.loc[result["rank"] <= top_n, "direction"] = "LONG"
    result.loc[result["rank"] > len(result) - top_n, "direction"] = "SHORT"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 출력 포매터
# ─────────────────────────────────────────────────────────────────────────────
def print_signal(signal_df: pd.DataFrame, top_n: int, market: str, horizon: int) -> None:
    log = logging.getLogger(__name__)
    date = signal_df["signal_date"].iloc[0]

    log.info(f"\n{'='*55}")
    log.info(f"  {market} 매매 신호 — {date}  (fwd{horizon}d 예측)")
    log.info(f"{'='*55}")

    long_df  = signal_df[signal_df["direction"] == "LONG"].head(top_n)
    short_df = signal_df[signal_df["direction"] == "SHORT"].tail(top_n)

    log.info(f"\n  📈 매수 상위 {top_n}종목")
    log.info(f"  {'순위':<5} {'티커':<8} {'종목명':<20} {'점수':>8}")
    log.info(f"  {'-'*45}")
    for _, row in long_df.iterrows():
        name = row.get("name", "")[:18] if "name" in row else ""
        log.info(f"  {int(row['rank']):<5} {row['ticker']:<8} {name:<20} {row['score']:>8.4f}")

    log.info(f"\n  📉 매도 하위 {top_n}종목")
    log.info(f"  {'순위':<5} {'티커':<8} {'종목명':<20} {'점수':>8}")
    log.info(f"  {'-'*45}")
    for _, row in short_df.sort_values("rank").iterrows():
        name = row.get("name", "")[:18] if "name" in row else ""
        log.info(f"  {int(row['rank']):<5} {row['ticker']:<8} {name:<20} {row['score']:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="일별 매매 신호 생성")
    parser.add_argument("--market",  choices=["KOSPI", "KOSDAQ", "ALL"], default="KOSPI")
    parser.add_argument("--horizon", choices=["5", "10", "20"], default="20")
    parser.add_argument("--mode",    choices=["ohlcv", "full"], default="full")
    parser.add_argument("--topn",    type=int, default=20, help="상위/하위 N종목")
    parser.add_argument("--no-download", action="store_true",
                        help="다운로드 스킵 (캐시 사용)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    log = logging.getLogger(__name__)

    load_env(override=True)
    markets = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]
    horizon = int(args.horizon)

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    for market in markets:
        try:
            # 유니버스 로드
            univ_path = RESULTS_DIR / "diagnose" / f"universe_filter_{market.lower()}.csv"
            universe = pd.read_csv(univ_path)["ticker"].tolist() if univ_path.exists() else None

            # 메타 로드
            meta_path = ROOT / "data" / "raw" / f"meta_{market.lower()}.parquet"
            meta = pd.read_parquet(meta_path) if meta_path.exists() else None

            # 최근 데이터
            if args.no_download:
                panel_path = ROOT / "data" / "raw" / f"{market.lower()}_panel.parquet"
                log.info(f"  캐시 패널 사용: {panel_path}")
                panel = pd.read_parquet(panel_path)
                # 최근 90일치만
                cutoff = panel.index.get_level_values("date").max() - pd.Timedelta(days=135)
                panel = panel[panel.index.get_level_values("date") >= cutoff]
            else:
                panel = fetch_recent(market, lookback_days=90)

            # 신호 생성
            signal_df = generate_signal(
                market, args.mode, horizon,
                args.topn, panel, universe, meta,
            )

            # 출력
            print_signal(signal_df, args.topn, market, horizon)

            # 저장
            today = datetime.today().strftime("%Y%m%d")
            save_path = SIGNALS_DIR / f"signal_{market.lower()}_{today}_fwd{horizon}.csv"
            signal_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            log.info(f"\n  💾 저장: {save_path}")

        except FileNotFoundError as e:
            log.error(f"\n  ❌ {e}")
        except Exception as e:
            log.exception(f"  ❌ {market} 신호 생성 실패: {e}")


if __name__ == "__main__":
    main()
