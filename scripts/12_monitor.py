"""
포지션 모니터링 — 20거래일 누적 신호 추적
=============================================

과거 20거래일치 신호 CSV를 모두 읽어서
각 신호일의 매수 상위 종목을 자동 추적.

- 신호일 종가 = 매수가 기준
- 20거래일 경과 시 기간 청산 알림
- 손절(-7%), 익절(+15%), 매도 신호 진입 알림

실행:
  python3 scripts/12_monitor.py
"""
from __future__ import annotations

import glob, logging, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

SIGNALS_DIR = ROOT / "results" / "signals"
RAW_DIR     = ROOT / "data" / "raw"

STOP_LOSS   = -0.07
TAKE_PROFIT =  0.15
HOLD_DAYS   = 20
LONG_TOPN   = 20

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def send_telegram(token, chat_id, text):
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"}, timeout=10)
        return r.status_code == 200
    except: return False


def count_trading_days(start_date_str: str) -> int:
    """신호일로부터 오늘까지 거래일 수."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    today = datetime.today()
    days, cur = 0, start
    while cur < today:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def load_active_signals() -> list[dict]:
    """
    과거 20거래일 이내의 신호 CSV 로드.
    각 신호 날짜별 매수 상위 종목 반환.
    """
    files = sorted(glob.glob(str(SIGNALS_DIR / "signal_kospi_*.csv")), reverse=True)
    if not files:
        return []

    active = []
    for f in files:
        # 파일명에서 날짜 추출
        stem = Path(f).stem  # signal_kospi_20260614_fwd20
        parts = stem.split("_")
        if len(parts) < 3: continue
        date_str = parts[2]  # YYYYMMDD
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        # 20거래일 초과된 신호는 스킵
        hold = count_trading_days(date_fmt)
        if hold > HOLD_DAYS:
            break  # 더 오래된 파일은 볼 필요 없음

        try:
            df = pd.read_csv(f)
            df["ticker"] = df["ticker"].astype(str).str.zfill(6)
            long_df = df[df["direction"] == "LONG"].head(LONG_TOPN)

            active.append({
                "signal_date": date_fmt,
                "hold_days":   hold,
                "long_df":     long_df,
                "all_df":      df,
            })
        except Exception as e:
            log.warning(f"  신호 파일 로드 실패 {f}: {e}")

    log.info(f"  활성 신호: {len(active)}개 날짜 ({len(active)*LONG_TOPN}개 포지션)")
    return active


def get_buy_prices(tickers: list[str], signal_date: str) -> dict[str, float]:
    """신호일 종가 조회 (매수가 기준)."""
    price_map = {}
    for path in [RAW_DIR/"kospi_recent.parquet", RAW_DIR/"kospi_panel.parquet"]:
        if not path.exists(): continue
        try:
            panel = pd.read_parquet(path, columns=["close"])
            dates = panel.index.get_level_values("date")
            target = pd.to_datetime(signal_date)

            # 정확한 날짜 또는 가장 가까운 이전 날짜
            avail = pd.DatetimeIndex(dates.unique()).sort_values()
            close_dates = avail[avail <= target]
            if not len(close_dates): continue
            use_date = close_dates[-1]

            day_data = panel[dates == use_date]
            for tkr in tickers:
                if tkr in price_map: continue
                try:
                    price_map[tkr] = float(
                        day_data.loc[(slice(None), tkr), "close"].iloc[0]
                    )
                except: pass

            if len(price_map) >= len(tickers): break
        except Exception as e:
            log.warning(f"  패널 로드 실패: {e}")
    return price_map


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """오늘 현재가 조회."""
    try:
        from data.env_loader import load_env; load_env(override=True)
        from pykrx import stock
        today = datetime.today().strftime("%Y%m%d")
        prices = {}
        for tkr in tickers:
            try:
                df = stock.get_market_ohlcv(today, today, tkr)
                if df is not None and not df.empty:
                    prices[tkr] = float(df["종가"].iloc[-1])
                time.sleep(0.15)
            except: pass
        return prices
    except Exception as e:
        log.error(f"  현재가 조회 실패: {e}"); return {}


def get_latest_short_tickers() -> set[str]:
    """오늘 최신 신호의 매도 종목 조회."""
    files = sorted(glob.glob(str(SIGNALS_DIR / "signal_kospi_*.csv")), reverse=True)
    if not files: return set()
    try:
        df = pd.read_csv(files[0])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return set(df[df["direction"]=="SHORT"]["ticker"].tolist())
    except: return set()


def run_monitor(token, chat_id):
    today_str = datetime.today().strftime("%Y-%m-%d")
    active_signals = load_active_signals()

    if not active_signals:
        log.info("  추적할 신호 없음"); return

    # 모든 활성 종목 수집 (중복 허용 — 날짜별로 별도 추적)
    all_tickers = list({
        tkr
        for sig in active_signals
        for tkr in sig["long_df"]["ticker"].tolist()
    })

    log.info(f"  현재가 조회 중 ({len(all_tickers)}종목)...")
    cur_prices   = get_current_prices(all_tickers)
    short_tickers = get_latest_short_tickers()

    # ── 날짜별 분석 ──
    alerts       = []
    date_sections = []
    grand_pnl_sum = 0
    grand_count   = 0

    for sig in active_signals:
        signal_date = sig["signal_date"]
        hold_days   = sig["hold_days"]
        long_df     = sig["long_df"]
        tickers     = long_df["ticker"].tolist()

        buy_prices = get_buy_prices(tickers, signal_date)

        rows = []
        section_pnl = []

        for _, row in long_df.iterrows():
            tkr  = row["ticker"]
            name = str(row.get("name", tkr))[:8]
            buy  = buy_prices.get(tkr)
            cur  = cur_prices.get(tkr)

            if not buy or not cur:
                continue

            pnl_pct  = (cur - buy) / buy
            icon     = "📈" if pnl_pct >= 0 else "📉"
            in_short = tkr in short_tickers
            section_pnl.append(pnl_pct)
            grand_pnl_sum += pnl_pct
            grand_count   += 1

            rows.append(
                f"  {icon} <b>{name}</b>({tkr})  {pnl_pct:+.1%}"
                + (f" ⚠️매도진입" if in_short else "")
            )

            # 알림 생성
            if pnl_pct <= STOP_LOSS:
                alerts.append(
                    f"🚨 <b>손절 알림</b> [{signal_date}] — {name}({tkr})\n"
                    f"매수기준 {buy:,.0f}원 → 현재 {cur:,.0f}원\n"
                    f"수익률 <b>{pnl_pct:+.1%}</b> | 보유 {hold_days}거래일\n"
                    f"⚡ 즉시 매도 권고"
                )
            elif pnl_pct >= TAKE_PROFIT:
                alerts.append(
                    f"✅ <b>익절 알림</b> [{signal_date}] — {name}({tkr})\n"
                    f"매수기준 {buy:,.0f}원 → 현재 {cur:,.0f}원\n"
                    f"수익률 <b>{pnl_pct:+.1%}</b> | 보유 {hold_days}거래일\n"
                    f"💡 절반 익절 권고"
                )
            if in_short:
                alerts.append(
                    f"⚠️ <b>매도 신호 진입</b> [{signal_date}] — {name}({tkr})\n"
                    f"현재 수익률 {pnl_pct:+.1%} | 보유 {hold_days}거래일\n"
                    f"💡 {'손실 중 → 즉시 청산 검토' if pnl_pct < 0 else '수익 중 → 익절 후 청산 검토'}"
                )
            if hold_days >= HOLD_DAYS:
                alerts.append(
                    f"⏰ <b>기간 청산 알림</b> [{signal_date}] — {name}({tkr})\n"
                    f"보유 {hold_days}거래일 (20거래일 도달)\n"
                    f"현재 수익률 {pnl_pct:+.1%}\n"
                    f"💡 청산 후 새 신호로 교체"
                )

        if rows:
            avg = sum(section_pnl)/len(section_pnl) if section_pnl else 0
            remain = max(0, HOLD_DAYS - hold_days)
            date_sections.append(
                f"📅 <b>{signal_date}</b> ({hold_days}거래일 경과 | 잔여 {remain}일 | 평균 {avg:+.1%})\n"
                + "\n".join(rows)
            )

    # ── 전체 요약 메시지 ──
    grand_avg = grand_pnl_sum / grand_count if grand_count else 0
    msg = (
        f"📊 <b>누적 신호 추적</b> ({today_str})\n"
        f"추적 기간: 최근 {HOLD_DAYS}거래일 | 총 {grand_count}포지션\n"
        f"전체 평균 수익률: <b>{grand_avg:+.1%}</b>\n\n"
        + "\n\n".join(date_sections)
    )

    if token and chat_id:
        # 메시지가 길면 분할 발송
        if len(msg) > 3800:
            chunks = [msg[i:i+3800] for i in range(0, len(msg), 3800)]
            for chunk in chunks:
                send_telegram(token, chat_id, chunk)
                time.sleep(0.5)
        else:
            send_telegram(token, chat_id, msg)

        for a in alerts:
            time.sleep(1)
            send_telegram(token, chat_id, a)
        log.info(f"  ✅ 발송 완료 (알림 {len(alerts)}개)")
    else:
        log.info("\n" + msg)
        for a in alerts: log.info("\n⚠️\n" + a)


if __name__ == "__main__":
    from data.env_loader import load_env; load_env(override=True)
    token   = os.environ.get("TELEGRAM_TOKEN","")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID","")
    run_monitor(token, chat_id)