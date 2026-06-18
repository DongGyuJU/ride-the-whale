"""
스케줄러 — 매일 자동 실행
==============================

평일 오후 4시 00분: 데이터 업데이트 (10_update_data.py)
평일 오후 4시 30분: 신호 생성 + 텔레그램 발송 (09_telegram.py)

실행 (백그라운드):
  nohup python3 scripts/11_scheduler.py &

중지:
  kill $(cat /root/smart_money/logs/scheduler.pid)
"""
import schedule
import subprocess
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "scheduler.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def run(script: str, args: list[str] = []) -> None:
    cmd = ["python3", str(ROOT / "scripts" / script)] + args
    log.info(f"실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if result.returncode == 0:
        log.info(f"✅ {script} 완료")
    else:
        log.error(f"❌ {script} 실패 (returncode={result.returncode})")


def job_update():
    log.info("="*50)
    log.info("  데이터 업데이트 시작")
    run("10_update_data.py", ["--market", "KOSPI"])

def job_earnings_update():
    """어닝 데이터 분기별 업데이트 (실적 발표 후)."""
    log.info("="*50)
    log.info("  어닝 데이터 업데이트 (DART API)")
    run("fetch_earnings.py")


def job_macro_update():
    """매크로 데이터 (금리 ETF) 주간 업데이트."""
    log.info("="*50)
    log.info("  매크로 데이터 업데이트")
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        from data.env_loader import load_env; load_env(override=True)
        from pykrx import stock
        import pandas as pd
        from datetime import datetime, timedelta
        import time as t
        MACRO_DIR = ROOT / "data" / "macro"
        end   = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=450)).strftime("%Y%m%d")
        for ticker, name in [("148070","bond_10y"),("114820","bond_3y"),("069500","kospi_index")]:
            df = stock.get_market_ohlcv(start, end, ticker)
            df.index = pd.to_datetime(df.index); df.index.name = "date"
            df.to_parquet(MACRO_DIR / f"{name}.parquet", compression="snappy")
            log.info(f"  매크로 {name}: {df.index.max().date()}")
            t.sleep(0.5)
    except Exception as e:
        log.error(f"  매크로 업데이트 실패: {e}")


def job_monitor():
    log.info("="*50)
    log.info("  포지션 모니터링 (손절/익절/매도진입 체크)")
    run("12_monitor.py", ["--run"])


def job_signal():
    log.info("="*50)

    # 1. 270종목 top20
    log.info("  [1/2] 일반 유니버스 (270종목) top20")
    run("09_telegram.py", [
        "--market", "KOSPI",
        "--topn", "20",
        "--capital", "10000000",
        "--score-threshold", "2.0",
        "--max-price", "1000000",
    ])

    time.sleep(10)

    # 2. KOSPI200 top10
    log.info("  [2/2] KOSPI200 top10")
    run("09_telegram.py", [
        "--market", "KOSPI",
        "--topn", "10",
        "--capital", "10000000",
        "--score-threshold", "2.0",
        "--max-price", "1000000",
        "--universe", "kospi200",
    ])

def is_weekday() -> bool:
    return datetime.today().weekday() < 5  # 0=월 ~ 4=금


def guarded(job_fn):
    """평일만 실행."""
    def wrapper():
        if is_weekday():
            job_fn()
        else:
            log.info(f"주말 — {job_fn.__name__} 스킵")
    return wrapper


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)

    # PID 저장 (kill 용도)
    pid_path = ROOT / "logs" / "scheduler.pid"
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))

    log.info(f"스케줄러 시작 (PID={os.getpid()})")
    log.info("  평일 16:00 — 데이터 업데이트")
    log.info("  평일 16:30 — 신호 + 텔레그램 발송")

    # 한국 시간 기준 (서버가 UTC면 -9시간)
    # UTC 서버: 07:00, 07:30 / KST 서버: 16:00, 16:30
    import subprocess as sp
    tz = sp.run(["date", "+%Z"], capture_output=True, text=True).stdout.strip()
    log.info(f"  서버 타임존: {tz}")

    if tz in ("KST", "Asia/Seoul"):
        update_time = "16:00"
        signal_time = "16:30"
    else:
        # UTC 기준
        update_time = "07:00"
        signal_time = "07:30"

    log.info(f"  업데이트 시간: {update_time} ({tz})")
    log.info(f"  발송 시간:     {signal_time} ({tz})")

    if tz in ("KST", "Asia/Seoul"):
        monitor_time = "16:10"
    else:
        monitor_time = "07:10"
    macro_time = "07:05" if tz not in ("KST","Asia/Seoul") else "16:05"
    schedule.every().day.at(update_time).do(guarded(job_update))
    schedule.every().monday.at(macro_time).do(guarded(job_macro_update))
    # 어닝: 3/6/9/12월 15일 실적 발표 후 자동 수집
    from datetime import datetime as dt
    if dt.today().month in [3, 6, 9, 12] and dt.today().day == 15:
        schedule.every().day.at("08:00").do(guarded(job_earnings_update))
    log.info("  어닝 업데이트: 3/6/9/12월 15일 자동")
    schedule.every().day.at(monitor_time).do(guarded(job_monitor))
    schedule.every().day.at(signal_time).do(guarded(job_signal))
    log.info(f"  모니터링 시간:   {monitor_time} ({tz})")
    log.info(f"  매크로 업데이트: 매주 월요일 {macro_time} ({tz})")

    # 시작 시 즉시 테스트 실행 옵션
    if "--run-now" in sys.argv:
        log.info("  --run-now: 즉시 실행")
        job_update()
        job_signal()

    log.info("  대기 중...")
    while True:
        schedule.run_pending()
        time.sleep(30)