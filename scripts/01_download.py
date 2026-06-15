"""
Stage 1-A: KRX 전체 시장 다운로드
=====================================
KOSPI + KOSDAQ, 2018-01-01 ~ 2024-12-31

출력:
  cache/KOSPI/  ── 종목별 .parquet (캐시)
  cache/KOSDAQ/ ── 종목별 .parquet (캐시)
  data/raw/kospi_panel.parquet   ── KOSPI 패널 (MultiIndex: date × ticker)
  data/raw/kosdaq_panel.parquet  ── KOSDAQ 패널
  data/raw/meta_kospi.parquet    ── KOSPI 메타
  data/raw/meta_kosdaq.parquet   ── KOSDAQ 메타

실행:
  # Docker
  docker compose run --rm cli python3 scripts/01_download.py

  # 직접 실행 (프로젝트 루트에서)
  python3 scripts/01_download.py [--market KOSPI|KOSDAQ|ALL] [--workers N] [--force]

⚠️  첫 실행은 KRX 서버 부하 고려해 workers=2 권장.
    자정~새벽 실행 시 차단 가능성 낮음.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 (어디서 실행해도 동작)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.env_loader import load_env
from data.krx_loader import load_market

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = "20180101"
END_DATE   = "20241231"
CACHE_DIR  = ROOT / "cache"
RAW_DIR    = ROOT / "data" / "raw"

MARKET_CONFIG = {
    "KOSPI":  {"workers": 2, "sleep_between": 0.3},
    "KOSDAQ": {"workers": 2, "sleep_between": 0.3},
}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ROOT / "logs" / "01_download.log", mode="a"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel 빌더
# ─────────────────────────────────────────────────────────────────────────────
def build_panel(
    meta_df: pd.DataFrame,
    ticker_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    {ticker: df} → MultiIndex (date, ticker) 패널 DataFrame.

    컬럼:
        open, high, low, close, volume, trade_value
        retail_net, foreign_net, inst_net, other_net
    """
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
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()

    # market_cap 붙이기 (메타에서)
    if "market_cap" in meta_df.columns:
        cap_map = meta_df.set_index("ticker")["market_cap"].to_dict()
        panel["market_cap"] = panel.index.get_level_values("ticker").map(cap_map)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 단일 시장 다운로드
# ─────────────────────────────────────────────────────────────────────────────
def download_market(
    market: str,
    workers: int,
    force: bool,
) -> None:
    log = logging.getLogger(__name__)
    cfg = MARKET_CONFIG[market]
    w = min(workers, cfg["workers"]) if workers else cfg["workers"]

    log.info("=" * 60)
    log.info(f"  {market} 다운로드 시작")
    log.info(f"  기간: {START_DATE} ~ {END_DATE}")
    log.info(f"  workers: {w}")
    log.info("=" * 60)

    t0 = time.time()
    meta_df, ticker_map = load_market(
        start_date=START_DATE,
        end_date=END_DATE,
        market=market,
        cache_dir=str(CACHE_DIR),
        max_workers=w,
        force_refresh=force,
    )
    elapsed = time.time() - t0

    log.info(f"  다운로드 완료: {len(ticker_map)}/{len(meta_df)} 종목  "
             f"({elapsed/60:.1f}분 소요)")

    # 패널 빌드 & 저장
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"  패널 빌드 중...")
    panel = build_panel(meta_df, ticker_map)

    if panel.empty:
        log.error(f"  ❌ 패널 비어있음 — KRX_ID/PW 확인 필요")
        return

    panel_path = RAW_DIR / f"{market.lower()}_panel.parquet"
    meta_path  = RAW_DIR / f"meta_{market.lower()}.parquet"

    panel.to_parquet(panel_path, compression="snappy")
    meta_df.to_parquet(meta_path, compression="snappy")

    # 요약 출력
    dates = panel.index.get_level_values("date").unique()
    tickers = panel.index.get_level_values("ticker").unique()
    supply_ok = panel["foreign_net"].notna().mean() if "foreign_net" in panel.columns else 0.0

    log.info(f"\n  ✅ {market} 저장 완료")
    log.info(f"     패널: {panel_path}")
    log.info(f"     날짜 범위: {dates.min().date()} ~ {dates.max().date()} ({len(dates)}거래일)")
    log.info(f"     유효 종목: {len(tickers)}")
    log.info(f"     외인 수급 비율(non-NaN): {supply_ok:.1%}")
    log.info(f"     Panel 크기: {panel.shape[0]:,} rows × {panel.shape[1]} cols")
    log.info(f"     파일 크기: {panel_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KRX KOSPI/KOSDAQ 수급+OHLCV 다운로드"
    )
    parser.add_argument(
        "--market",
        choices=["KOSPI", "KOSDAQ", "ALL"],
        default="ALL",
        help="다운로드할 시장 (기본값: ALL)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="동시 다운로드 수 (0=자동, 권장: 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시 무시하고 강제 재다운로드",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 로그 디렉토리 먼저
    (ROOT / "logs").mkdir(exist_ok=True)
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    # 환경변수 로드
    # ℹ️  pykrx는 KRX 공개 API를 스크래핑 → 로그인 불필요.
    #    KRX_ID/PW는 현재 코드에서 실제로 사용되지 않음.
    #    (향후 프리미엄 데이터 확장 대비 보존)
    load_env(override=True)
    log.info("pykrx 사용 — KRX 로그인 불필요 (공개 API)")

    markets = ["KOSPI", "KOSDAQ"] if args.market == "ALL" else [args.market]

    total_start = time.time()
    for market in markets:
        try:
            download_market(market, args.workers, args.force)
        except Exception as e:
            log.exception(f"❌ {market} 다운로드 실패: {e}")
            log.info("  캐시는 보존됨 — 재실행 시 이어받기 가능")

    total_elapsed = time.time() - total_start
    log.info(f"\n전체 소요: {total_elapsed/60:.1f}분")
    log.info("다음 단계: python3 scripts/02_diagnose.py")


if __name__ == "__main__":
    main()
