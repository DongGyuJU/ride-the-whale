"""
B. 연속 신호 추적
==================

매일 신호 CSV를 비교해서 연속으로 상위권에 든 종목 표시.
같은 종목이 연속 N일 상위권 → 신뢰도 상승.

결과: results/signals/consecutive_YYYYMMDD.csv
"""
from __future__ import annotations

import glob, logging, os, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

SIGNALS_DIR = ROOT / "results" / "signals"
LONG_TOPN   = 20

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def send_telegram(token, chat_id, text):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"}, timeout=10)
    except: pass


def run(token="", chat_id="", lookback=5):
    """최근 lookback일간 연속 신호 분석."""
    files = sorted(glob.glob(str(SIGNALS_DIR / "signal_kospi_*.csv")), reverse=True)
    if len(files) < 2:
        log.info("  신호 파일 2개 이상 필요")
        return

    # 최근 lookback개 파일 로드
    recent = files[:lookback]
    log.info(f"  분석 기간: {len(recent)}일치 신호")

    # 날짜별 상위 종목 집계
    date_longs: dict[str, set] = {}
    date_scores: dict[str, dict] = {}

    for f in recent:
        stem = Path(f).stem
        date_str = stem.split("_")[2]
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        df = pd.read_csv(f)
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        long_df = df[df["direction"]=="LONG"].head(LONG_TOPN)

        date_longs[date_fmt]  = set(long_df["ticker"].tolist())
        date_scores[date_fmt] = dict(zip(long_df["ticker"], long_df["score"]))

    dates = sorted(date_longs.keys(), reverse=True)
    today = dates[0]

    # 오늘 상위 종목 중 연속 등장 횟수 계산
    today_longs = date_longs[today]
    consecutive: dict[str, int] = {}

    for tkr in today_longs:
        count = 0
        for d in dates:
            if tkr in date_longs[d]:
                count += 1
            else:
                break
        consecutive[tkr] = count

    # 결과 정리
    today_df = pd.read_csv(recent[0])
    today_df["ticker"] = today_df["ticker"].astype(str).str.zfill(6)
    today_df = today_df[today_df["direction"]=="LONG"].head(LONG_TOPN).copy()
    today_df["consecutive"] = today_df["ticker"].map(consecutive)
    today_df = today_df.sort_values("consecutive", ascending=False)

    # 결과 저장
    save_path = SIGNALS_DIR / f"consecutive_{today.replace('-','')}.csv"
    today_df[["ticker","name","score","consecutive"]].to_csv(save_path, index=False)

    # 텔레그램 발송
    multi = today_df[today_df["consecutive"] >= 3]

    if not multi.empty:
        lines = [
            f"🔁 <b>연속 신호 종목</b> ({today})",
            f"최근 {lookback}일 기준 3일↑ 연속 상위권 진입\n",
        ]
        for _, r in multi.iterrows():
            stars = "⭐" * min(int(r["consecutive"]), 5)
            lines.append(
                f"  {stars} <b>{r['name'][:8]}</b>({r['ticker']})  "
                f"{int(r['consecutive'])}일 연속  "
                f"<code>{r['score']:+.3f}</code>"
            )
        lines.append(f"\n<i>연속 등장 = 신호 일관성 ↑ = 신뢰도 높음</i>")
        msg = "\n".join(lines)

        if token and chat_id:
            send_telegram(token, chat_id, msg)
            log.info(f"  ✅ 연속 신호 {len(multi)}종목 발송")
        else:
            log.info("\n" + msg)
    else:
        log.info("  3일 이상 연속 신호 종목 없음")

    # 전체 출력
    log.info(f"\n  연속 신호 현황 (오늘 top{LONG_TOPN}):")
    for _, r in today_df.iterrows():
        bar = "█" * int(r["consecutive"])
        log.info(f"  {r['name'][:10]:<12} {bar} {int(r['consecutive'])}일")

    return today_df


if __name__ == "__main__":
    from data.env_loader import load_env
    load_env(override=True)
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    run(token, chat_id)
