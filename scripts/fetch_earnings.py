"""
서버에서 실행: DART 어닝 데이터 수집
======================================

DART OpenAPI로 KOSPI 유니버스 종목의
분기별 영업이익/순이익 수집.

실행:
  python3 scripts/fetch_earnings.py

출력:
  data/earnings/kospi_earnings.parquet
"""
import os, sys, time, requests, zipfile, io
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.env_loader import load_env
load_env(override=True)

import pandas as pd

DART_KEY = os.environ.get("DART_API_KEY", "")
SAVE_DIR = ROOT / "data" / "earnings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 수집 분기 범위
YEARS    = list(range(2018, datetime.today().year + 1))
QUARTERS = {
    "11011": ("Q1", "03-31"),
    "11012": ("Q2", "06-30"),
    "11013": ("Q3", "09-30"),
    "11014": ("Q4", "12-31"),
}

# 공시 후 평균 45일 뒤 시장에 반영 (미래정보 방지)
REPORT_DELAY_DAYS = 45


def get_corp_code_map() -> dict:
    """DART 기업코드 ↔ 주식코드 매핑."""
    print("  DART 기업코드 매핑 다운로드 중...")
    r = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": DART_KEY}, timeout=30
    )
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    corp_map = {}  # stock_code → corp_code
    for item in tree.getroot():
        stock_code = item.findtext("stock_code", "").strip().zfill(6)
        corp_code  = item.findtext("corp_code", "").strip()
        if stock_code and corp_code and stock_code != "000000":
            corp_map[stock_code] = corp_code

    print(f"  매핑: {len(corp_map)}개 종목")
    return corp_map


def fetch_income(corp_code: str, year: int, reprt_code: str) -> dict | None:
    """분기 손익 데이터 조회."""
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS",  # 연결재무제표
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "000":
            return None

        result = {}
        for item in data.get("list", []):
            nm  = item.get("account_nm", "")
            val = item.get("thstrm_amount", "").replace(",", "").replace(" ", "")
            try:
                v = float(val) if val else None
            except:
                v = None

            if "영업이익" in nm and "영업이익률" not in nm:
                result["op_income"] = v
            elif "당기순이익" in nm and "지배" in nm:
                result["net_income"] = v
            elif "매출액" in nm or "수익(매출액)" in nm:
                result["revenue"] = v

        return result if result else None

    except Exception as e:
        return None


def main():
    if not DART_KEY:
        print("❌ DART_API_KEY 없음. .env에 추가 필요")
        return

    print(f"DART 어닝 데이터 수집 시작")
    print(f"  대상: {YEARS[0]}~{YEARS[-1]} / 분기별")

    # 유니버스 로드
    univ_path = ROOT / "results" / "diagnose" / "universe_filter_kospi.csv"
    if not univ_path.exists():
        print("❌ 유니버스 파일 없음")
        return
    universe = pd.read_csv(univ_path)["ticker"].astype(str).str.zfill(6).tolist()
    print(f"  유니버스: {len(universe)}종목")

    # 기업코드 매핑
    corp_map = get_corp_code_map()
    target = {tkr: corp_map[tkr] for tkr in universe if tkr in corp_map}
    print(f"  DART 매핑 성공: {len(target)}종목")

    # 분기별 수집
    rows = []
    total = len(target) * len(YEARS) * len(QUARTERS)
    done  = 0
    errors = 0

    for tkr, corp_code in target.items():
        for year in YEARS:
            for reprt_code, (q_label, q_end) in QUARTERS.items():
                done += 1
                if done % 200 == 0:
                    print(f"  진행: {done}/{total} | 수집: {len(rows)} | 오류: {errors}")

                result = fetch_income(corp_code, year, reprt_code)
                if result:
                    # 공시 날짜 = 분기말 + 45일 (미래정보 방지)
                    fiscal_end   = pd.Timestamp(f"{year}-{q_end}")
                    report_date  = fiscal_end + pd.Timedelta(days=REPORT_DELAY_DAYS)
                    rows.append({
                        "ticker":         tkr,
                        "year":           year,
                        "quarter":        q_label,
                        "fiscal_quarter": f"{year}{q_label}",
                        "fiscal_end":     fiscal_end,
                        "report_date":    report_date,
                        **result,
                    })
                else:
                    errors += 1

                time.sleep(0.08)  # DART API 속도 제한

    if not rows:
        print("❌ 수집된 데이터 없음")
        return

    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.sort_values(["ticker", "report_date"]).reset_index(drop=True)

    # 변화율 계산
    df["op_yoy"]    = df.groupby(["ticker","quarter"])["op_income"].pct_change()
    df["op_qoq"]    = df.groupby("ticker")["op_income"].pct_change()
    df["op_accel"]  = df.groupby("ticker")["op_qoq"].diff()  # 2차 미분
    df["rev_yoy"]   = df.groupby(["ticker","quarter"])["revenue"].pct_change()

    save_path = SAVE_DIR / "kospi_earnings.parquet"
    df.to_parquet(save_path, compression="snappy", index=False)

    print(f"\n✅ 저장: {save_path}")
    print(f"   행수: {len(df)}")
    print(f"   종목: {df['ticker'].nunique()}")
    print(f"   기간: {df['fiscal_quarter'].min()} ~ {df['fiscal_quarter'].max()}")
    print(f"\n샘플 (삼성전자):")
    sam = df[df["ticker"]=="005930"][["fiscal_quarter","op_income","op_yoy","op_accel"]].tail(6)
    print(sam.to_string())


if __name__ == "__main__":
    main()
