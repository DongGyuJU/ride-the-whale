"""
H. 실제 IC 자동 검증 시스템
==============================

매일 실행:
  1. 20거래일 전 신호 로드
  2. 오늘 가격으로 실제 수익률 계산
  3. 신호 점수 vs 실제 수익률 IC 계산
  4. 누적 IC 트래킹 + 텔레그램 알림

파일:
  results/ic_tracker/daily_ic.csv     — 일별 IC 기록
  results/ic_tracker/summary.json     — 요약 통계

실행:
  python3 scripts/13_ic_tracker.py
  python3 scripts/13_ic_tracker.py --report  # 전체 보고서
"""
from __future__ import annotations

import argparse, glob, json, logging, os, sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

SIGNALS_DIR = ROOT / "results" / "signals"
IC_DIR      = ROOT / "results" / "ic_tracker"
RAW_DIR     = ROOT / "data" / "raw"
IC_DIR.mkdir(parents=True, exist_ok=True)

HOLD_DAYS = 20  # fwd20

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def count_trading_days(start: str, end: str) -> int:
    """두 날짜 사이 거래일 수 (주말 제외)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    days, cur = 0, s
    while cur < e:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def get_price_map(date_str: str) -> dict[str, float]:
    """특정 날짜 종가 조회."""
    for path in [RAW_DIR/"kospi_recent.parquet", RAW_DIR/"kospi_panel.parquet"]:
        if not path.exists():
            continue
        try:
            panel = pd.read_parquet(path, columns=["close"])
            dates = panel.index.get_level_values("date")
            target = pd.to_datetime(date_str)
            avail  = pd.DatetimeIndex(dates.unique()).sort_values()
            close  = avail[avail <= target]
            if not len(close):
                continue
            use_date = close[-1]
            day_data = panel[dates == use_date]
            result = {}
            for (_, tkr), row in day_data.iterrows():
                result[str(tkr).zfill(6)] = float(row["close"])
            if result:
                return result
        except Exception as e:
            log.warning(f"  가격 로드 실패 {path}: {e}")
    return {}


def compute_daily_ic(signal_file: str) -> dict | None:
    """
    신호 파일 하나에 대한 실제 IC 계산.
    20거래일 후 가격이 있을 때만 계산.
    """
    stem  = Path(signal_file).stem           # signal_kospi_20260529_fwd20
    parts = stem.split("_")
    if len(parts) < 3:
        return None

    date_str  = parts[2]
    sig_date  = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    today_str = datetime.today().strftime("%Y-%m-%d")

    # 20거래일 지났는지 확인
    hold = count_trading_days(sig_date, today_str)
    if hold < HOLD_DAYS:
        return None

    try:
        sig_df = pd.read_csv(signal_file)
        sig_df["ticker"] = sig_df["ticker"].astype(str).str.zfill(6)
        sig_df = sig_df[sig_df["direction"].isin(["LONG","SHORT","neutral"])]

        if len(sig_df) < 20:
            return None

        # 신호일 가격
        price_sig = get_price_map(sig_date)
        # 오늘 가격
        price_now = get_price_map(today_str)

        rows = []
        for _, r in sig_df.iterrows():
            tkr = r["ticker"]
            p0  = price_sig.get(tkr)
            p1  = price_now.get(tkr)
            if not p0 or not p1:
                continue
            ret = (p1 - p0) / p0
            rows.append({"ticker": tkr, "score": r["score"], "return": ret})

        if len(rows) < 20:
            return None

        df = pd.DataFrame(rows).dropna()
        ic_val = spearmanr(df["score"], df["return"]).statistic
        t_stat = ic_val * np.sqrt(len(df)) / np.sqrt(1 - ic_val**2 + 1e-10)

        return {
            "signal_date": sig_date,
            "eval_date":   today_str,
            "hold_days":   hold,
            "n_stocks":    len(df),
            "ic":          round(ic_val, 4),
            "t_stat":      round(t_stat, 2),
            "long_ret":    round(df[df["score"] > 0]["return"].mean(), 4),
            "short_ret":   round(df[df["score"] < 0]["return"].mean(), 4),
            "long_short":  round(df[df["score"] > 0]["return"].mean()
                               - df[df["score"] < 0]["return"].mean(), 4),
        }

    except Exception as e:
        log.warning(f"  IC 계산 실패 {signal_file}: {e}")
        return None


def load_ic_history() -> pd.DataFrame:
    ic_file = IC_DIR / "daily_ic.csv"
    if ic_file.exists():
        return pd.read_csv(ic_file)
    return pd.DataFrame(columns=["signal_date","eval_date","hold_days",
                                  "n_stocks","ic","t_stat","long_ret",
                                  "short_ret","long_short"])


def save_ic_history(df: pd.DataFrame) -> None:
    df.to_csv(IC_DIR / "daily_ic.csv", index=False)


def compute_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ic_series = df["ic"]
    return {
        "n_signals":    len(df),
        "mean_ic":      round(ic_series.mean(), 4),
        "std_ic":       round(ic_series.std(), 4),
        "z_stat":       round(ic_series.mean() / (ic_series.std() / np.sqrt(len(df)) + 1e-10), 2),
        "ic_positive":  round((ic_series > 0).mean(), 3),
        "mean_ls_ret":  round(df["long_short"].mean(), 4),
        "last_ic":      round(ic_series.iloc[-1], 4) if len(df) else None,
        "updated":      datetime.today().strftime("%Y-%m-%d"),
        "status": (
            "✅ 유효" if ic_series.mean() > 0.03 else
            "⚠️ 약화" if ic_series.mean() > 0.01 else
            "❌ 재학습 필요"
        )
    }


def send_telegram(token, chat_id, text):
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML"}, timeout=10)
    except: pass


def run(token="", chat_id=""):
    log.info("IC 검증 시작...")

    files = sorted(glob.glob(str(SIGNALS_DIR / "signal_kospi_*.csv")))
    history = load_ic_history()
    computed = set(history["signal_date"].tolist()) if not history.empty else set()

    new_rows = []
    for f in files:
        result = compute_daily_ic(f)
        if result and result["signal_date"] not in computed:
            new_rows.append(result)
            log.info(f"  {result['signal_date']} IC={result['ic']:+.4f} "
                     f"t={result['t_stat']:.2f} LS={result['long_short']:+.2%}")

    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)
        history = history.sort_values("signal_date").reset_index(drop=True)
        save_ic_history(history)

    summary = compute_summary(history)
    with open(IC_DIR / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not summary:
        log.info("  아직 20거래일 경과한 신호 없음 (최소 한 달 후 확인)")
        return

    log.info(f"\n  평균 IC: {summary['mean_ic']:+.4f}")
    log.info(f"  Z-stat:  {summary['z_stat']:.2f}")
    log.info(f"  IC>0:    {summary['ic_positive']:.1%}")
    log.info(f"  상태:    {summary['status']}")

    # 텔레그램 알림 (상태 변화 시)
    if token and chat_id:
        msg = (
            f"📈 <b>모델 IC 검증 보고서</b>\n"
            f"검증 신호: {summary['n_signals']}개\n\n"
            f"평균 IC: <b>{summary['mean_ic']:+.4f}</b>\n"
            f"Z-stat:  {summary['z_stat']:.2f}\n"
            f"IC>0:    {summary['ic_positive']:.1%}\n"
            f"L-S수익: {summary['mean_ls_ret']:+.2%}\n\n"
            f"상태: {summary['status']}\n\n"
            f"<i>IC>0.03 유효 | IC<0.01 재학습 필요</i>"
        )
        send_telegram(token, chat_id, msg)


def print_report():
    history = load_ic_history()
    if history.empty:
        print("아직 검증 데이터 없음")
        return

    print(f"\n{'='*55}")
    print(f"  모델 IC 검증 보고서")
    print(f"{'='*55}")
    print(f"  {'신호일':<12} {'IC':>7} {'t-stat':>7} {'L-S수익':>8} {'종목수':>6}")
    print(f"  {'-'*50}")
    for _, r in history.iterrows():
        status = "✅" if r["ic"] > 0.03 else "⚠️" if r["ic"] > 0 else "❌"
        print(f"  {r['signal_date']:<12} {r['ic']:>+7.4f} {r['t_stat']:>7.2f} "
              f"{r['long_short']:>+8.2%} {r['n_stocks']:>6}  {status}")

    summary = compute_summary(history)
    print(f"\n  {'─'*50}")
    print(f"  평균 IC:  {summary['mean_ic']:+.4f}")
    print(f"  Z-stat:   {summary['z_stat']:.2f}")
    print(f"  IC>0:     {summary['ic_positive']:.1%}")
    print(f"  L-S수익:  {summary['mean_ls_ret']:+.2%}")
    print(f"  상태:     {summary['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="전체 보고서 출력")
    args = parser.parse_args()

    from data.env_loader import load_env
    load_env(override=True)
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if args.report:
        print_report()
    else:
        run(token, chat_id)
