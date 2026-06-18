"""
로컬(맥)에서 실행: KOSPI 공매도 데이터 수집
=============================================

날짜별 전체 시장 공매도 조회 → 합쳐서 parquet 저장.
월 1회 실행 권장.

실행:
  python3 scripts/local_fetch_short.py

출력:
  data/short/kospi_short.parquet
"""
import sys, time
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from data.env_loader import load_env
    load_env(override=True)
except: pass

from pykrx import stock
import pandas as pd

SAVE_DIR = ROOT / "data" / "short"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

START = "20200101"
END   = datetime.today().strftime("%Y%m%d")


def get_trading_days(start: str, end: str) -> list[str]:
    """거래일 목록 생성 (주말 제외, 공휴일은 데이터 없으면 스킵)."""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end,   "%Y%m%d")
    days = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:  # 월~금
            days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days


def fetch_short_data():
    print(f"KOSPI 공매도 데이터 수집: {START} ~ {END}")
    days = get_trading_days(START, END)
    print(f"  대상 거래일: {len(days)}일")

    frames = []
    failed = 0

    for i, day in enumerate(days):
        if i % 100 == 0:
            print(f"  진행: {i}/{len(days)} | 성공: {len(frames)} | 실패: {failed}")

        try:
            df = stock.get_shorting_volume_by_ticker(day, "KOSPI")
            if df is None or df.empty:
                failed += 1
                continue

            # 컬럼 정리
            df = df.rename(columns={
                "공매도": "short_volume",
                "매수":   "total_volume",
                "비중":   "short_ratio_pct",
            })
            df.index.name = "ticker"
            df.index = df.index.astype(str).str.zfill(6)
            df["date"] = pd.to_datetime(day)
            df["short_ratio"] = df["short_ratio_pct"] / 100  # % → 소수

            frames.append(df.reset_index().set_index(["date", "ticker"]))
            time.sleep(0.2)

        except Exception as e:
            failed += 1
            if "None of" not in str(e):  # ISU_CD 에러는 무시
                pass

    if not frames:
        print("❌ 데이터 없음")
        return

    panel = pd.concat(frames).sort_index()

    # 유니버스 필터
    univ_path = ROOT / "results" / "diagnose" / "universe_filter_kospi.csv"
    if univ_path.exists():
        universe = set(pd.read_csv(univ_path)["ticker"].astype(str).str.zfill(6).tolist())
        tickers  = panel.index.get_level_values("ticker")
        panel    = panel[tickers.isin(universe)]
        print(f"  유니버스 필터 후: {panel.index.get_level_values('ticker').nunique()}종목")

    save_path = SAVE_DIR / "kospi_short.parquet"
    panel.to_parquet(save_path, compression="snappy")

    print(f"\n✅ 저장: {save_path}")
    print(f"   크기: {panel.shape}")
    print(f"   기간: {panel.index.get_level_values('date').min().date()} ~ "
          f"{panel.index.get_level_values('date').max().date()}")
    print(f"   성공: {len(frames)}일 | 실패: {failed}일")
    print(f"\n구글드라이브 업로드:")
    print(f"  MyDrive/project/smart_money/data/short/kospi_short.parquet")


if __name__ == "__main__":
    fetch_short_data()