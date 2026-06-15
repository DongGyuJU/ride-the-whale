"""
KRX 통합 데이터 로더.

기능:
- 종목 메타 (티커, 이름, 시장(KOSPI/KOSDAQ), 시총)
- OHLCV (일봉)
- 매매주체별 수급 (개인/외국인/기관/기타법인, 거래대금 단위)

설계:
- 시장(market) 인자로 KOSPI / KOSDAQ 분리 다운로드 가능
- 각 ticker별로 OHLCV와 수급을 join → 단일 DataFrame
- 캐시 파일명에 시장 + 기간 포함 (충돌 방지)
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Literal
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MarketType = Literal['KOSPI', 'KOSDAQ', 'ALL']


# ─────────────────────────────────────────────────────────────────────────────
# Cache utilities
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    """병렬 쓰기 안전한 parquet 저장 (tmp → rename)."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_parquet(tmp, compression='snappy')
    tmp.replace(path)


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Meta (종목 리스트 + 시장 + 시총 + 섹터)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_meta(
    end_date: str,
    market: MarketType = 'KOSPI',
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    종목 메타 (티커, 이름, 시장, 시총).
    시총은 종목별 개별 조회 대신 시장 전체 한 번에 가져옴 (KRX 차단 방지).

    Returns:
        DataFrame[ticker, name, market, market_cap]
    """
    if cache_path:
        cached = _safe_read_parquet(cache_path)
        if cached is not None and len(cached) > 0:
            logger.info(f"[meta] 캐시 사용: {len(cached)} 종목 ({market})")
            return cached

    from pykrx import stock

    markets = ['KOSPI', 'KOSDAQ'] if market == 'ALL' else [market]

    all_rows = []
    for mkt in markets:
        tickers = stock.get_market_ticker_list(end_date, market=mkt)
        logger.info(f"[meta] {mkt}: {len(tickers)} 종목 이름 수집 중...")

        # 이름 수집
        names = {}
        for tkr in tickers:
            try:
                names[tkr] = stock.get_market_ticker_name(tkr)
            except Exception:
                names[tkr] = tkr

        # 시총 — 시장 전체 한 번에 (종목별 개별 조회 X → KRX 차단 방지)
        logger.info(f"[meta] {mkt}: 시총 일괄 조회 중...")
        mcap_map = {}
        try:
            cap_df = stock.get_market_cap_by_date(end_date, end_date, mkt)
            # 반환 형태: index=date, columns에 ticker별 시총 또는
            # get_market_cap_by_ticker 사용
        except Exception:
            cap_df = None

        # get_market_cap_by_ticker: 특정 날짜 전 종목 시총 한방에
        try:
            cap_ticker_df = stock.get_market_cap_by_ticker(end_date, market=mkt)
            if cap_ticker_df is not None and not cap_ticker_df.empty:
                col = '시가총액' if '시가총액' in cap_ticker_df.columns else cap_ticker_df.columns[0]
                mcap_map = cap_ticker_df[col].to_dict()
                logger.info(f"[meta] {mkt}: 시총 일괄 조회 완료 ({len(mcap_map)} 종목)")
        except Exception as e:
            logger.warning(f"[meta] {mkt}: 시총 일괄 조회 실패 ({e}) → 0으로 채움")

        for tkr in tickers:
            all_rows.append({
                'ticker': tkr,
                'name': names.get(tkr, tkr),
                'market': mkt,
                'market_cap': int(mcap_map.get(tkr, 0)),
            })

    df = pd.DataFrame(all_rows)
    if cache_path:
        _atomic_write_parquet(df, cache_path)
    logger.info(f"[meta] 완료: {len(df)} 종목")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Single-ticker OHLCV + supply
# ─────────────────────────────────────────────────────────────────────────────
def fetch_one_ohlcv(ticker: str, start: str, end: str,
                     retries: int = 3) -> Optional[pd.DataFrame]:
    """단일 종목 OHLCV."""
    from pykrx import stock
    for attempt in range(retries):
        try:
            df = stock.get_market_ohlcv(start, end, ticker)
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                '시가': 'open', '고가': 'high', '저가': 'low',
                '종가': 'close', '거래량': 'volume', '거래대금': 'trade_value',
                '등락률': 'pct_change',
            })
            if 'trade_value' not in df.columns and 'close' in df.columns:
                df['trade_value'] = df['close'] * df['volume']
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                logger.debug(f"  {ticker} OHLCV 실패: {e}")
    return None


def fetch_one_supply(ticker: str, start: str, end: str,
                      retries: int = 3) -> Optional[pd.DataFrame]:
    """
    단일 종목 매매 주체별 일별 순매수 (거래대금 기준, 단위: 원).

    Returns:
        DataFrame[retail_net, foreign_net, inst_net, other_net]
    """
    from pykrx import stock
    for attempt in range(retries):
        try:
            df = stock.get_market_trading_value_by_date(
                start, end, ticker, detail=False,
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                '개인': 'retail_net',
                '외국인': 'foreign_net',
                '외국인합계': 'foreign_net',
                '기관': 'inst_net',
                '기관합계': 'inst_net',
                '기타법인': 'other_net',
            })
            cols = ['retail_net', 'foreign_net', 'inst_net', 'other_net']
            df = df[[c for c in cols if c in df.columns]]
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                logger.debug(f"  {ticker} 수급 실패: {e}")
    return None


def fetch_one_complete(
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """
    OHLCV + 수급 한 번에 받아서 join.

    Returns:
        DataFrame[open, high, low, close, volume, trade_value,
                  retail_net, foreign_net, inst_net, other_net]
    """
    ohlcv = fetch_one_ohlcv(ticker, start, end)
    if ohlcv is None or ohlcv.empty:
        return None
    supply = fetch_one_supply(ticker, start, end)
    if supply is not None:
        df = ohlcv.join(supply, how='left')
    else:
        df = ohlcv.copy()
        for c in ['retail_net', 'foreign_net', 'inst_net', 'other_net']:
            df[c] = np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Bulk loader
# ─────────────────────────────────────────────────────────────────────────────
def load_market(
    start_date: str,
    end_date: str,
    market: MarketType = 'KOSPI',
    cache_dir: str = 'cache',
    max_workers: int = 4,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    전체 시장 다운로드: 메타 + 종목별 OHLCV+수급.

    Args:
        start_date, end_date: 'YYYYMMDD'
        market: 'KOSPI' | 'KOSDAQ' | 'ALL'
        cache_dir: 캐시 디렉토리

    Returns:
        (meta_df, ohlcv_supply_map)
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    market_cache = cache / market
    market_cache.mkdir(exist_ok=True)
    data_cache = market_cache / f'data_{start_date}_{end_date}'
    data_cache.mkdir(exist_ok=True)

    # 1. 메타
    meta_path = market_cache / f'meta_{end_date}.parquet'
    if force_refresh:
        meta_path.unlink(missing_ok=True)
    meta_df = fetch_meta(end_date, market=market, cache_path=meta_path)

    if meta_df.empty or 'ticker' not in meta_df.columns:
        raise RuntimeError(f"[load_market] 메타 비어있음 ({market}).")

    tickers = meta_df['ticker'].tolist()

    # 2. 종목별 OHLCV+수급
    def _load_one(tkr):
        f = data_cache / f'{tkr}.parquet'
        if not force_refresh:
            cached = _safe_read_parquet(f)
            if cached is not None:
                if not isinstance(cached.index, pd.DatetimeIndex):
                    cached.index = pd.to_datetime(cached.index)
                return tkr, cached
        df = fetch_one_complete(tkr, start_date, end_date)
        if df is not None and len(df) > 0:
            _atomic_write_parquet(df, f)
        return tkr, df

    logger.info(f"[load_market] {market}: {len(tickers)} 종목 다운로드 시작 "
                f"(workers={max_workers})")
    result_map = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_one, t): t for t in tickers}
        for fut in as_completed(futures):
            tkr, df = fut.result()
            if df is not None and not df.empty:
                result_map[tkr] = df
            completed += 1
            if completed % 100 == 0:
                logger.info(f"  진행: {completed}/{len(tickers)} (성공 {len(result_map)})")

    logger.info(f"[load_market] {market} 완료: {len(result_map)}/{len(tickers)} 종목")
    return meta_df, result_map


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import sys
    sys.path.insert(0, '..')
    from data.env_loader import load_env
    load_env(override=True)

    print("Samsung Electronics (005930) 테스트:")
    df = fetch_one_complete('005930', '20240101', '20240131')
    if df is not None:
        print(df.head())
        print(f"\n컬럼: {list(df.columns)}")
    else:
        print("❌ 다운로드 실패")
