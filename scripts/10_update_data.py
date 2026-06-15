"""
매일 실행: 최근 90일치 데이터 업데이트
=========================================

전체 패널 재다운로드 없이 최근 90일치만 받아서
data/raw/kospi_recent.parquet 에 저장.

09_telegram.py가 이 파일을 우선 사용.

실행:
  python3 scripts/10_update_data.py --market KOSPI
  
소요 시간: 약 5~10분 (종목별 딜레이 포함)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

RAW_DIR   = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "cache" / "daily_recent"


def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])


def fetch_recent_panel(market: str, lookback_days: int = 90) -> pd.DataFrame:
    """
    최근 lookback_days 거래일치 OHLCV + 수급 다운로드.
    딜레이를 충분히 줘서 KRX 차단 방지.
    """
    # pykrx import 전에 반드시 load_env 호출 (KRX 로그인 환경변수 설정)
    from data.env_loader import load_env
    load_env(override=True)
    from pykrx import stock
    log = logging.getLogger(__name__)

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.6))
    end_str   = end_dt.strftime("%Y%m%d")
    start_str = start_dt.strftime("%Y%m%d")

    log.info(f"  기간: {start_str} ~ {end_str}")

    # 종목 리스트 — 유니버스 CSV 사용
    # KRX 시장 전체 API는 차단됨. 종목별 OHLCV만 가능.
    # 신규 상장 종목은 60일치 과거 없으면 피처 계산 불가 → 월 1회 수동 갱신으로 충분.
    univ_path = ROOT/"results"/"diagnose"/f"universe_filter_{market.lower()}.csv"
    if not univ_path.exists():
        raise RuntimeError("유니버스 파일 없음. 먼저 02_diagnose.py 실행 필요")

    tickers = pd.read_csv(univ_path)["ticker"].astype(str).str.zfill(6).tolist()
    log.info(f"  유니버스 종목: {len(tickers)}개 (월 1회 02_diagnose.py로 갱신)")

    # 종목별 다운로드
    frames = []
    failed = []
    total = len(tickers)

    for i, tkr in enumerate(tickers):
        if i % 50 == 0:
            log.info(f"  진행: {i}/{total} (성공 {len(frames)})")

        try:
            # OHLCV
            ohlcv = stock.get_market_ohlcv(start_str, end_str, tkr)
            if ohlcv is None or ohlcv.empty:
                failed.append(tkr)
                time.sleep(0.3)
                continue

            ohlcv = ohlcv.rename(columns={
                '시가':'open','고가':'high','저가':'low',
                '종가':'close','거래량':'volume','거래대금':'trade_value',
                '등락률':'pct_change',
            })
            # trade_value 없으면 close * volume 으로 계산
            if 'trade_value' not in ohlcv.columns:
                ohlcv['trade_value'] = ohlcv['close'] * ohlcv['volume']
            ohlcv.index = pd.to_datetime(ohlcv.index)
            ohlcv.index.name = "date"

            # 수급 — 실패해도 OHLCV만으로 계속 진행
            time.sleep(0.2)
            try:
                supply = stock.get_market_trading_value_by_date(
                    start_str, end_str, tkr, detail=False
                )
                if supply is not None and not supply.empty:
                    supply = supply.rename(columns={
                        '개인':'retail_net','외국인':'foreign_net',
                        '외국인합계':'foreign_net','기관':'inst_net',
                        '기관합계':'inst_net','기타법인':'other_net',
                    })
                    supply.index = pd.to_datetime(supply.index)
                    supply.index.name = "date"
                    cols = [c for c in ['retail_net','foreign_net','inst_net','other_net']
                            if c in supply.columns]
                    ohlcv = ohlcv.join(supply[cols], how='left')
            except Exception:
                # 수급 실패 → NaN으로 채움 (OHLCV 모델은 수급 없어도 동작)
                pass

            ohlcv["ticker"] = tkr
            frames.append(ohlcv)

        except Exception as e:
            failed.append(tkr)

        # 딜레이 — KRX 차단 방지 (0.15 → 0.4초)
        time.sleep(0.4)

    if not frames:
        raise RuntimeError("다운로드된 데이터 없음")

    log.info(f"  완료: {len(frames)}/{total} 성공 | 실패: {len(failed)}")
    if failed[:5]:
        log.info(f"  실패 예시: {failed[:5]}")

    panel = pd.concat(frames).reset_index()

    # 티커를 문자열 6자리로 통일 (int로 저장되는 경우 방지)
    panel["ticker"] = panel["ticker"].astype(str).str.zfill(6)
    panel["date"]   = pd.to_datetime(panel["date"])

    panel = panel.set_index(["date", "ticker"]).sort_index()

    latest = panel.index.get_level_values("date").max()
    log.info(f"  최신 날짜: {latest.date()}")
    log.info(f"  티커 샘플: {panel.index.get_level_values('ticker').unique()[:3].tolist()}")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["KOSPI","KOSDAQ","ALL"], default="KOSPI")
    parser.add_argument("--lookback", type=int, default=90)
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    markets = ["KOSPI","KOSDAQ"] if args.market == "ALL" else [args.market]

    for market in markets:
        log.info(f"\n{'='*50}")
        log.info(f"  {market} 최근 데이터 업데이트")
        log.info(f"{'='*50}")
        try:
            t0 = time.time()
            panel = fetch_recent_panel(market, args.lookback)

            save_path = RAW_DIR / f"{market.lower()}_recent.parquet"
            panel.to_parquet(save_path, compression="snappy")
            elapsed = time.time() - t0
            log.info(f"  ✅ 저장: {save_path}")
            log.info(f"  크기: {panel.shape} | 소요: {elapsed/60:.1f}분")

        except Exception as e:
            log.exception(f"  ❌ {market} 업데이트 실패: {e}")

    log.info("\n다음 단계: python3 scripts/09_telegram.py")


if __name__ == "__main__":
    main()
