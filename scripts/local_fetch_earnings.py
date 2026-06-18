"""
로컬(맥)에서 실행: DART 어닝 데이터 수집
==========================================

DART OpenAPI로 분기 실적 데이터 수집.
분기 1회 실행 권장 (실적 발표 후 2주 내).

사전 준비:
  1. dart.fss.or.kr 가입 후 API 키 발급 (무료)
  2. .env에 DART_API_KEY=your_key 추가

실행:
  python3 scripts/local_fetch_earnings.py

출력:
  data/earnings/kospi_earnings.parquet
"""
import os, sys, time, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from data.env_loader import load_env
    load_env(override=True)
except:
    pass

import pandas as pd

DART_API_KEY = os.environ.get("DART_API_KEY", "")
SAVE_DIR = ROOT / "data" / "earnings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# DART 재무항목 코드
ACCOUNT_NM = "영업이익"  # 또는 "당기순이익"

def get_dart_corp_codes() -> pd.DataFrame:
    """DART 기업 고유번호 조회."""
    url = "https://opendart.fss.or.kr/api/company.json"
    # 유니버스 종목 로드
    univ = pd.read_csv(ROOT/"results"/"diagnose"/"universe_filter_kospi.csv")
    tickers = univ["ticker"].astype(str).str.zfill(6).tolist()
    print(f"대상 종목: {len(tickers)}개")
    return tickers


def fetch_quarter_income(corp_code: str, year: int, quarter: int) -> dict | None:
    """분기 손익계산서 조회."""
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": f"1100{quarter}",  # 11011=1Q, 11012=2Q, 11013=3Q, 11014=4Q
        "fs_div": "CFS",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "000":
            return None
        for item in data.get("list", []):
            if item.get("account_nm") == ACCOUNT_NM:
                return {
                    "op_income": float(item.get("thstrm_amount", "0").replace(",","")),
                    "year": year,
                    "quarter": quarter,
                }
    except:
        pass
    return None


def fetch_all_earnings():
    if not DART_API_KEY:
        print("❌ DART_API_KEY 없음. .env에 추가 후 재실행")
        print("   dart.fss.or.kr 에서 무료 발급")
        return

    tickers = get_dart_corp_codes()

    # DART 기업코드 매핑 (stock_code → corp_code)
    print("DART 기업코드 매핑 중...")
    corp_map_url = "https://opendart.fss.or.kr/api/corpCode.xml"
    # 간단히 corp_code.xml 다운로드 후 파싱
    r = requests.get(f"https://opendart.fss.or.kr/api/corpCode.xml",
                     params={"crtfc_key": DART_API_KEY}, timeout=30)

    import zipfile, io, xml.etree.ElementTree as ET
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    corp_map = {}
    for item in tree.getroot():
        stock_code = item.findtext("stock_code", "").strip()
        corp_code  = item.findtext("corp_code", "").strip()
        if stock_code and corp_code:
            corp_map[stock_code.zfill(6)] = corp_code

    print(f"매핑된 종목: {len(corp_map)}개")

    # 최근 8분기 실적 수집
    from datetime import datetime
    current_year = datetime.today().year
    quarters = []
    for y in range(current_year-1, current_year+1):
        for q in [1, 2, 3, 4]:
            quarters.append((y, q))

    frames = []
    for i, tkr in enumerate(tickers[:50]):  # 테스트는 50개
        corp_code = corp_map.get(tkr)
        if not corp_code:
            continue

        if i % 10 == 0:
            print(f"  진행: {i}/{min(50, len(tickers))}")

        for year, quarter in quarters[-8:]:  # 최근 8분기
            result = fetch_quarter_income(corp_code, year, quarter)
            if result:
                result["ticker"]      = tkr
                result["report_date"] = f"{year}-{'03-31' if quarter==1 else '06-30' if quarter==2 else '09-30' if quarter==3 else '12-31'}"
                result["fiscal_quarter"] = f"{year}Q{quarter}"
                frames.append(result)
            time.sleep(0.15)

    if not frames:
        print("데이터 없음")
        return

    df = pd.DataFrame(frames)
    df["report_date"] = pd.to_datetime(df["report_date"]) + pd.Timedelta(days=45)  # 공시 지연 45일 가산
    save_path = SAVE_DIR / "kospi_earnings.parquet"
    df.to_parquet(save_path, compression="snappy", index=False)
    print(f"\n✅ 저장: {save_path} ({len(df)}행)")
    print("구글드라이브: MyDrive/project/smart_money/data/earnings/")


if __name__ == "__main__":
    fetch_all_earnings()
