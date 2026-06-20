"""
Stage 7: 텔레그램 알림 (앙상블 지원)
=======================================

단일 모델 + ensemble3 pkl 모두 지원.

.env 필요:
  TELEGRAM_TOKEN=your_bot_token
  TELEGRAM_CHAT_ID=your_chat_id

실행:
  # 앙상블 (기본)
  python3 scripts/09_telegram.py --market KOSPI --capital 2000000 --topn 30 --no-download

  # 단일 모델 지정
  python3 scripts/09_telegram.py --model-name ridge_kospi_ohlcv_fwd20 --no-download

cron (평일 8:30):
  30 8 * * 1-5 cd /root/smart_money && python3 scripts/09_telegram.py --no-download >> logs/telegram.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from data.env_loader import load_env
from features.technical import add_technical_features
from features.supply import add_supply_features
from features.pipeline import cross_sectional_zscore

SIGNALS_DIR = ROOT / "results" / "signals"
RAW_DIR     = ROOT / "data" / "raw"
MODEL_DIR   = ROOT / "models" / "saved"


def setup_logging():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])


# ─────────────────────────────────────────────────────────────────────────────
# 텔레그램
# ─────────────────────────────────────────────────────────────────────────────
def send_message(token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logging.getLogger(__name__).error(f"  발송 실패: {e}")
        return False


def send_long(token: str, chat_id: str, text: str) -> bool:
    log = logging.getLogger(__name__)
    # 줄 단위로 청크 분할 (HTML 태그 중간에서 잘리지 않게)
    lines = text.split("\n")
    chunks, cur = [], ""
    for line in lines:
        if len(cur) + len(line) + 1 > 4000:
            if cur: chunks.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur: chunks.append(cur)
    for i, chunk in enumerate(chunks):
        try:
            # 1차: HTML 모드
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code == 200:
                time.sleep(0.3)
                continue
            # HTML 실패 → 에러 로깅
            err = resp.json().get("description", "unknown")
            log.warning(f"  HTML 발송 실패: {err}")
            # 2차: plain text로 재시도 (HTML 태그 제거)
            import re
            plain = re.sub(r"<[^>]+>", "", chunk)
            resp2 = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": plain},
                timeout=10,
            )
            if resp2.status_code != 200:
                log.error(f"  plain text도 실패: {resp2.json()}")
                return False
            time.sleep(0.3)
        except Exception as e:
            log.error(f"  발송 예외: {e}")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 패널 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_panel(market: str, no_download: bool) -> pd.DataFrame:
    """
    패널 로드 우선순위:
    1. _recent.parquet (10_update_data.py 로 매일 갱신)
    2. _panel.parquet (전체 히스토리, 최근 90일만 사용)
    """
    log = logging.getLogger(__name__)

    recent_path = RAW_DIR / f"{market.lower()}_recent.parquet"
    full_path   = RAW_DIR / f"{market.lower()}_panel.parquet"

    if recent_path.exists():
        panel = pd.read_parquet(recent_path)
        latest = panel.index.get_level_values("date").max()
        log.info(f"  최신 데이터 사용: {recent_path.name} (최신: {latest.date()})")

        # 오늘과 너무 차이나면 경고
        from datetime import datetime
        days_old = (datetime.today() - latest.to_pydatetime()).days
        if days_old > 5:
            log.warning(f"  ⚠️  데이터가 {days_old}일 오래됨 → 10_update_data.py 실행 권장")
        return panel

    elif full_path.exists():
        log.warning(f"  _recent.parquet 없음 → 전체 패널에서 최근 90일 사용")
        log.warning(f"  ⚠️  10_update_data.py 를 먼저 실행하면 오늘 날짜 신호 생성 가능")
        panel = pd.read_parquet(full_path)
        latest = panel.index.get_level_values("date").max()
        cutoff = latest - pd.Timedelta(days=135)
        panel = panel[panel.index.get_level_values("date") >= cutoff]
        log.info(f"  전체 패널 최신 날짜: {latest.date()}")
        return panel

    else:
        msg = f"패널 파일 없음. 먼저 실행: python3 scripts/10_update_data.py --market {market}"
        raise FileNotFoundError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 피처 계산
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(panel: pd.DataFrame, feat_cols: list[str],
                     universe: list[str] | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    log = logging.getLogger(__name__)
    if universe:
        tickers = panel.index.get_level_values("ticker")
        panel = panel[tickers.isin(universe)]

    # v2 피처 (tech_v2 + supply_v2)가 필요한지 확인
    has_v2_feats = any(c.startswith(("tech_52w","tech_bb","tech_tv","tech_macd",
                                      "tech_mom20","tech_vol20","tech_mom60"))
                       for c in feat_cols)
    try:
        if has_v2_feats:
            from features.technical_v2 import add_technical_features_v2
            from features.supply_v2 import add_supply_features_v2
            df = add_technical_features_v2(panel)
            if any(c in panel.columns for c in ["foreign_net","inst_net"]):
                df = add_supply_features_v2(df)
        else:
            df = add_technical_features(panel)
            if any(c in df.columns for c in ["foreign_net","inst_net"]):
                df = add_supply_features(df)
    except Exception:
        df = add_technical_features(panel)
        if any(c in df.columns for c in ["foreign_net","inst_net"]):
            df = add_supply_features(df)

    # 매크로 피처 추가 (v4/v5/v6 모델용)
    macro_needed = [c for c in feat_cols if c.startswith("macro_")]
    if macro_needed:
        try:
            from features.macro import load_macro, add_macro_features
            macro_dir = ROOT / "data" / "macro"
            if macro_dir.exists():
                macro_df = load_macro(macro_dir)
                df = add_macro_features(df, macro_df)
                log.info(f"  매크로 피처 추가: {len(macro_needed)}개")
        except Exception as e:
            log.warning(f"  매크로 피처 로드 실패: {e} → 0으로 채움")

    # 교호작용 피처 추가 (v5/v6 모델용)
    ix_needed = [c for c in feat_cols if c.startswith("ix_")]
    if ix_needed:
        try:
            rate = df["macro_rate_regime"] if "macro_rate_regime" in df.columns else None
            bull = df["macro_kospi_regime"] if "macro_kospi_regime" in df.columns else None
            for target in ["tech_52w_pos","tech_bb_pos","tech_tv_ratio",
                           "tech_mom20","tech_vol20","tech_mom60",
                           "sup_fmom_chg","sup_fnet_rank","sup_fcum20","sup_fstreak"]:
                if target not in df.columns: continue
                if rate is not None:
                    df[f"ix_rate_{target}"] = rate * df[target]
                if bull is not None:
                    df[f"ix_bull_{target}"] = bull * df[target]
            log.info(f"  교호작용 피처 생성 완료")
        except Exception as e:
            log.warning(f"  교호작용 피처 생성 실패: {e} → 0으로 채움")

    # 미분 피처 추가 (v6 모델용)
    d_needed = [c for c in feat_cols if c[:2] in ("d5","d2") and "_" in c]
    if d_needed:
        try:
            from features.derivatives import add_derivative_features
            df, _ = add_derivative_features(df, list(df.columns))
        except Exception as e:
            log.warning(f"  미분 피처 생성 실패: {e} → 0으로 채움")

    avail = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(avail)
    if missing:
        log.warning(f"  누락 피처 {len(missing)}개 → 0으로 채움")
        for m in missing:
            df[m] = 0.0
        avail = feat_cols

    feat_df = cross_sectional_zscore(df[avail], avail)
    latest_date = feat_df.index.get_level_values("date").max()
    latest = feat_df.loc[latest_date]
    log.info(f"  피처 날짜: {latest_date.date()} | 종목: {len(latest)}")
    return latest, latest_date


# ─────────────────────────────────────────────────────────────────────────────
# 예측 (단일 / 앙상블)
# ─────────────────────────────────────────────────────────────────────────────
def detect_regime(macro_df) -> str:
    """현재 레짐 감지: A/B/C/D"""
    if macro_df is None or macro_df.empty:
        return "A"
    latest = macro_df.iloc[-1]
    bull = latest.get("macro_kospi_regime", 0) > 0.5
    fall = latest.get("macro_rate_regime", 0)  > 0.5
    if bull and fall:   return "A"
    if bull and not fall: return "B"
    if not bull and fall: return "C"
    return "D"


REGIME_LABELS = {
    "A": "강세+금리하락 🟢",
    "B": "강세+금리상승 🟡",
    "C": "약세+금리하락 🟠",
    "D": "약세+금리상승 🔴",
}

REGIME_IC = {
    "A": 0.0967,
    "B": 0.0616,
    "C": 0.0069,
    "D": 0.0596,
}

# 레짐별 최적 호라이즌 (멀티 호라이즌 결과 기반)
REGIME_HORIZON = {
    "A": 60,   # 강세+금리하락: fwd60 IC=0.154
    "B": 60,   # 강세+금리상승: fwd60 IC=0.153
    "C": 20,   # 약세+금리하락: fwd5 음수, fwd20 사용
    "D": 20,   # 약세+금리상승: fwd5 음수, fwd20 사용
}

REGIME_IC_BEST = {
    "A": {"fwd5": 0.040, "fwd20": 0.097, "fwd60": 0.154},
    "B": {"fwd5": 0.050, "fwd20": 0.062, "fwd60": 0.153},
    "C": {"fwd5":-0.033, "fwd20": 0.011, "fwd60": 0.062},
    "D": {"fwd5":-0.025, "fwd20": 0.054, "fwd60": 0.063},
}


def predict_regime_v7(obj: dict, panel: pd.DataFrame,
                      universe: list | None,
                      macro_df=None) -> tuple:
    """레짐별 독립 모델로 예측 — 3개 호라이즌 동시."""
    log = logging.getLogger(__name__)
    import pickle as _pkl

    regime_id   = detect_regime(macro_df)
    regime_info = obj.get("regime_defs", {}).get(regime_id, {})
    use_fund    = regime_info.get("use_fundamental", False)
    opt_horizon = REGIME_HORIZON.get(regime_id, 20)

    log.info(f"  현재 레짐: {regime_id} ({REGIME_LABELS[regime_id]})")
    log.info(f"  레짐 IC: {REGIME_IC[regime_id]:.4f}")
    log.info(f"  최적 호라이즌: fwd{opt_horizon}")

    score_map   = {}   # {ticker: score} 메인
    h_score_map = {}   # {"fwd5": {ticker: score}, ...}
    signal_date = None

    for h in [5, 20, 60]:
        mpath = ROOT / "models" / "saved" / f"regime_v7_kospi_fwd{h}.pkl"
        if not mpath.exists():
            log.warning(f"  fwd{h} 모델 없음 — 스킵")
            continue
        with open(mpath, "rb") as mf:
            h_obj = _pkl.load(mf)

        h_info   = h_obj.get("regime_defs", {}).get(regime_id, {})
        h_fund   = h_info.get("use_fundamental", False)
        h_cols   = h_obj.get("feat_cols_fund", h_obj["feat_cols"]) if h_fund                    else h_obj["feat_cols"]
        h_model  = h_obj["regimes"].get(regime_id)
        if h_model is None:
            continue

        latest, sd = compute_features(panel, h_cols, universe)
        if signal_date is None:
            signal_date = sd

        # latest 인덱스 → ticker 문자열
        if isinstance(latest.index, pd.MultiIndex):
            tickers = latest.index.get_level_values("ticker").astype(str).str.zfill(6)
        else:
            tickers = latest.index.astype(str).str.zfill(6)

        avail = [c for c in h_cols if c in latest.columns]
        X = latest[avail].fillna(0).values
        X_s = h_model["scaler"].transform(X)
        s = 0.5 * h_model["ridge"].predict(X_s) + 0.5 * h_model["en"].predict(X_s)

        h_score_map[f"fwd{h}"] = dict(zip(tickers, s))
        if h == opt_horizon:
            score_map = dict(zip(tickers, s))
        log.info(f"  fwd{h} 예측 완료 ({len(tickers)}종목)")

    main_scores = pd.Series(score_map, name="score")
    return main_scores, h_score_map, signal_date, regime_id


def predict_single(obj: dict, panel: pd.DataFrame,
                   universe: list[str] | None) -> tuple[pd.Series, pd.Timestamp]:
    model     = obj["model"]
    scaler    = obj["scaler"]
    feat_cols = obj["feat_cols"]

    latest, signal_date = compute_features(panel, feat_cols, universe)
    avail = [c for c in feat_cols if c in latest.columns]
    X = latest[avail].fillna(0).values
    scores = model.predict(scaler.transform(X))
    return pd.Series(scores, index=latest.index, name="score"), signal_date


def predict_ensemble3(obj: dict, panel: pd.DataFrame,
                      universe: list[str] | None) -> tuple[pd.Series, pd.Timestamp]:
    log = logging.getLogger(__name__)
    w1 = obj["w_ridge_ohlcv"]
    w2 = obj["w_ridge_full"]
    w3 = obj["w_en_full"]

    log.info(f"  앙상블 가중치: ohlcv×{w1} + ridge_full×{w2} + en_full×{w3}")

    preds = {}
    signal_date = None

    for key, w in [("ridge_ohlcv", w1), ("ridge_full", w2), ("en_full", w3)]:
        if w == 0:
            log.info(f"  {key} 스킵 (가중치=0)")
            continue
        sub = obj[key]
        latest, sd = compute_features(panel, sub["feat_cols"], universe)
        avail = [c for c in sub["feat_cols"] if c in latest.columns]
        X = latest[avail].fillna(0).values
        pred = pd.Series(
            sub["model"].predict(sub["scaler"].transform(X)),
            index=latest.index,
        )
        # CS z-score 정규화 (스케일 통일)
        pred = (pred - pred.mean()) / (pred.std() + 1e-10)
        preds[key] = (pred, w)
        signal_date = sd
        log.info(f"  {key} 예측 완료 (가중치={w})")

    # 가중 합산
    all_tickers = preds[list(preds.keys())[0]][0].index
    ensemble = pd.Series(0.0, index=all_tickers)
    total_w = 0.0
    for key, (pred, w) in preds.items():
        common = ensemble.index.intersection(pred.index)
        ensemble.loc[common] += w * pred.loc[common]
        total_w += w
    ensemble = ensemble / (total_w + 1e-10)

    return ensemble, signal_date


def get_active_tickers(market: str) -> set[str] | None:
    """pykrx로 최근 거래일 기준 실제 상장 종목 조회."""
    log = logging.getLogger(__name__)
    try:
        from data.env_loader import load_env
        load_env(override=True)
        from pykrx import stock
        from datetime import datetime, timedelta
        import time
        # 오늘 장전이면 빈 응답 → 최근 거래일로 재시도
        for delta in range(0, 7):
            try_date = (datetime.today() - timedelta(days=delta)).strftime("%Y%m%d")
            tickers = stock.get_market_ticker_list(try_date, market=market)
            if tickers:
                active = set(tickers)
                log.info(f"  현재 상장 종목: {len(active)}개 ({market}, 기준: {try_date})")
                return active
            time.sleep(0.5)
        log.warning(f"  현존 종목 조회 실패 → 필터 스킵")
        return None
    except Exception as e:
        log.warning(f"  현존 종목 조회 실패: {e} → 필터 스킵")
        return None


def generate_signal(model_path: Path, panel: pd.DataFrame,
                    universe: list[str] | None,
                    market: str, horizon: int,
                    top_n: int) -> pd.DataFrame:
    log = logging.getLogger(__name__)

    # 현재 상장 종목과 교집합 (상장폐지 종목 자동 제거)
    active = get_active_tickers(market)
    if active and universe:
        before = len(universe)
        universe = [t for t in universe if t in active]
        removed = before - len(universe)
        if removed:
            log.info(f"  상장폐지/거래정지 제거: {removed}종목")
    elif active and not universe:
        universe = list(active)

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    model_type = obj.get("type", "single")
    log.info(f"  모델 타입: {model_type}")

    # 매크로 데이터 로드 (레짐 감지용)
    macro_df_for_regime = None
    try:
        from features.macro import load_macro
        macro_dir = ROOT / "data" / "macro"
        if macro_dir.exists():
            macro_df_for_regime = load_macro(macro_dir)
    except: pass

    if model_type == "regime_ensemble_v7":
        scores, h_scores, signal_date, regime_id = predict_regime_v7(
            obj, panel, universe, macro_df_for_regime
        )
        model_label = f"레짐 {regime_id} ({REGIME_LABELS.get(regime_id,'')}) IC={REGIME_IC.get(regime_id,0):.4f}"
    elif model_type.startswith("ensemble3"):
        scores, signal_date = predict_ensemble3(obj, panel, universe)
        model_label = (f"앙상블 (ohlcv×{obj['w_ridge_ohlcv']} "
                       f"rf×{obj['w_ridge_full']} en×{obj['w_en_full']})")
    else:
        scores, signal_date = predict_single(obj, panel, universe)
        model_label = model_path.stem

    # 메타 (종목명)
    meta_path = RAW_DIR / f"meta_{market.lower()}.parquet"
    name_map  = pd.read_parquet(meta_path).set_index("ticker")["name"].to_dict() \
                if meta_path.exists() else {}

    result = pd.DataFrame({
        "ticker": scores.index,
        "score":  scores.values,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    result["name"]        = result["ticker"].map(name_map).fillna("")
    result["signal_date"] = signal_date.date()
    result["market"]      = market
    result["horizon"]     = horizon
    result["model"]       = model_label
    result["rank"]        = range(1, len(result)+1)
    result["direction"]   = "neutral"
    result.loc[result["rank"] <= top_n,             "direction"] = "LONG"
    result.loc[result["rank"] > len(result)-top_n,  "direction"] = "SHORT"

    # 멀티 호라이즌 점수 병합
    if model_type == "regime_ensemble_v7":
        for h_key, h_map in h_scores.items():
            result[f"score_{h_key}"] = result["ticker"].map(h_map)

    return result


def filter_outliers(signal_df: pd.DataFrame, threshold: float, top_n: int) -> pd.DataFrame:
    """
    |score| > threshold 종목 제거 — 수급 데이터 없어서 피처 전부 0인 이상치.
    정상 CS z-score 범위는 ±2 이내. 기본 threshold=2.0
    """
    log = logging.getLogger(__name__)
    outliers = signal_df[signal_df["score"].abs() > threshold]
    if len(outliers):
        log.info(f"  이상치 제거 {len(outliers)}종목 (|score|>{threshold}):")
        for _, r in outliers.head(8).iterrows():
            log.info(f"    {r['ticker']} {r['name'][:10]:<10}  score={r['score']:.4f}")

    filtered = signal_df[signal_df["score"].abs() <= threshold].copy()
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
    filtered["rank"]      = range(1, len(filtered)+1)
    filtered["direction"] = "neutral"
    n = len(filtered)
    filtered.loc[filtered["rank"] <= top_n,     "direction"] = "LONG"
    filtered.loc[filtered["rank"] > n - top_n,  "direction"] = "SHORT"
    log.info(f"  필터 후: {n}종목 (제거 {len(outliers)}개)")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 포지션 사이징
# ─────────────────────────────────────────────────────────────────────────────
def calc_positions(signal_df: pd.DataFrame, capital: float,
                   market: str, max_price: float = 200_000) -> pd.DataFrame:
    long_df = signal_df[signal_df["direction"] == "LONG"].copy()

    panel = pd.read_parquet(RAW_DIR / f"{market.lower()}_panel.parquet")
    latest = panel.index.get_level_values("date").max()
    prices = {}
    for tkr in long_df["ticker"]:
        try:    prices[tkr] = float(panel.loc[(latest, tkr), "close"])
        except: prices[tkr] = 0

    long_df["price"] = long_df["ticker"].map(prices)

    # 고가주 + 가격 미확인 종목 제외
    if max_price > 0:
        excluded = long_df[(long_df["price"] > max_price) | (long_df["price"] <= 0)]
        if len(excluded):
            logging.getLogger(__name__).info(
                f"  고가주/미확인 제외 {len(excluded)}종목: "
                f"{excluded['ticker'].tolist()[:5]}{'...' if len(excluded)>5 else ''}"
            )
        long_df = long_df[(long_df["price"] > 0) & (long_df["price"] <= max_price)]

    n     = len(long_df)
    alloc = capital / n if n > 0 else 0
    rows  = []
    for _, row in long_df.iterrows():
        price  = row["price"]
        shares = int(alloc / price)
        if shares == 0:  # 배분금액보다 주가가 높은 경우 스킵
            continue
        rows.append({
            "ticker":     row["ticker"],
            "name":       row["name"],
            "score":      round(row["score"], 4),
            "weight":     round(1.0/n, 4),
            "price":      round(price),
            "shares":     shares,
            "alloc_krw":  round(alloc),
            "actual_krw": round(shares * price),
        })

    # 0주 제외 후 비중 재계산
    result = pd.DataFrame(rows)
    if len(result):
        result["weight"] = round(1.0 / len(result), 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 메시지 포매터
# ─────────────────────────────────────────────────────────────────────────────
def get_ticker_metrics(tickers: list) -> dict:
    """종목별 핵심 피처값 실시간 계산."""
    try:
        panel = pd.read_parquet(RAW_DIR / "kospi_recent.parquet")
        metrics = {}
        # panel ticker를 6자리 문자열로 정규화
        panel_idx = panel.index.get_level_values("ticker").astype(str).str.zfill(6)
        for tkr in tickers:
            tkr = str(tkr).zfill(6)
            try:
                tkr_data = panel[panel_idx == tkr].sort_index()
                if tkr_data.empty:
                    continue
                close = tkr_data["close"]
                fnet  = tkr_data["foreign_net"] if "foreign_net" in tkr_data.columns else pd.Series(dtype=float)
                tv    = tkr_data["trade_value"]  if "trade_value"  in tkr_data.columns else pd.Series(dtype=float)

                # 외인 매수 가속도 (최근 5일 vs 이전 5일)
                fmom_chg = None
                if len(fnet) >= 10 and len(tv) >= 10:
                    f5_cur  = fnet.iloc[-5:].sum()
                    f5_prev = fnet.iloc[-10:-5].sum()
                    tv10    = tv.iloc[-10:].sum()
                    if tv10 > 0:
                        fmom_chg = (f5_cur - f5_prev) / tv10 * 100

                # 외인 10일 누적 순매수
                fcum10 = None
                if len(fnet) >= 10 and len(tv) >= 10:
                    tv10 = tv.iloc[-10:].sum()
                    if tv10 > 0:
                        fcum10 = fnet.iloc[-10:].sum() / tv10 * 100

                # 52주 위치
                pos52w = None
                if len(close) >= 60:
                    mn = close.iloc[-252:].min() if len(close) >= 252 else close.min()
                    mx = close.iloc[-252:].max() if len(close) >= 252 else close.max()
                    if mx > mn:
                        pos52w = (close.iloc[-1] - mn) / (mx - mn) * 100

                # 20일 수익률
                mom20 = None
                if len(close) >= 21:
                    mom20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100

                # 기관 연속 매수일수
                istreak = 0
                inet = tkr_data["inst_net"] if "inst_net" in tkr_data.columns else pd.Series(dtype=float)
                if len(inet) >= 3:
                    for v in reversed(inet.iloc[-10:].values):
                        if pd.isna(v): break
                        if v > 0: istreak += 1
                        else: break

                # 기관 5일 누적
                icum5 = None
                if len(inet) >= 5 and len(tv) >= 5:
                    tv5 = tv.iloc[-5:].sum()
                    if tv5 > 0:
                        icum5 = max(-30, min(30, inet.iloc[-5:].sum() / tv5 * 100))

                metrics[tkr] = {
                    "fmom_chg": fmom_chg,
                    "fcum10":   fcum10,
                    "pos52w":   pos52w,
                    "mom20":    mom20,
                    "istreak":  istreak,
                    "icum5":    icum5,
                }
            except:
                pass
        return metrics
    except Exception as e:
        logging.getLogger(__name__).warning(f"  피처 계산 실패: {e}")
        return {}


def fmt_reason(tkr: str, score: float, rank: int, total: int,
               metrics: dict, macro_info: dict | None = None) -> tuple:
    """
    종목별 신호 이유 반환.
    Returns: (warning: bool, line1: str, line2: str)
    """
    m = metrics.get(tkr, {})
    fmom = m.get("fmom_chg")
    fcum = m.get("fcum10")
    p52  = m.get("pos52w")
    mom  = m.get("mom20")

    # 경고 조건 1: 외인 매도 중인데 상위권
    foreign_selling = (fmom is not None and fmom < -1.0) or                       (fcum is not None and fcum < -3.0)
    warning = foreign_selling and score > 0.5

    # 경고 조건 2: 페이딩 신호 (스파이크 후 식어가는 중)
    # fcum10 과도하게 높은데 fmom_chg 음수 = 과거 스파이크가 희석 중
    fading = (fcum is not None and fcum > 20.0 and
              fmom is not None and fmom < 0)
    if fading:
        warning = True

    # 라인1: 기술적 (52주 위치 + 모멘텀)
    tech_parts = []
    if p52 is not None:
        if p52 >= 80:
            tech_parts.append(f"📍52주 {p52:.0f}%")
        elif p52 >= 60:
            tech_parts.append(f"📍52주 {p52:.0f}%")
        else:
            tech_parts.append(f"📍52주 {p52:.0f}% (저점)")
    if mom is not None and abs(mom) >= 3:
        icon = "🔺" if mom > 0 else "🔻"
        tech_parts.append(f"{icon}20일 {mom:+.1f}%")

    istreak = m.get("istreak", 0)
    icum5   = m.get("icum5")

    # 라인2: 수급
    sup_parts = []
    if fmom is not None:
        if fmom > 0.5:
            sup_parts.append(f"🐋외인가속 +{fmom:.1f}%")
        elif fmom > 0:
            sup_parts.append(f"🐋외인↑ +{fmom:.1f}%")
        elif fmom < -0.5:
            sup_parts.append(f"🔴외인↓ {fmom:.1f}%")
    if fcum is not None and abs(fcum) >= 0.3:
        arrow = "↑" if fcum > 0 else "↓"
        color = "🟢" if fcum > 0 else "🔴"
        sup_parts.append(f"{color}외인10일 {fcum:+.1f}%{arrow}")
    if istreak >= 3:
        sup_parts.append(f"🏦기관{istreak}일연속↑")
    elif istreak <= -3:
        sup_parts.append(f"🏦기관{abs(istreak)}일연속↓")
    elif icum5 is not None and abs(icum5) >= 0.5:
        arrow = "↑" if icum5 > 0 else "↓"
        color = "🟢" if icum5 > 0 else "🔴"
        sup_parts.append(f"{color}기관5일 {icum5:+.1f}%{arrow}")

    # 레짐
    regime_parts = []
    if macro_info:
        if macro_info.get("kospi_regime", 0) > 0.5:
            regime_parts.append("강세장✓")
        if macro_info.get("rate_regime", 0) > 0.5:
            regime_parts.append("금리하락✓")

    line1 = " · ".join(tech_parts) if tech_parts else "기술적 신호"
    if fading:
        sup_parts.insert(0, "📉페이딩신호")
    line2 = " · ".join(sup_parts + regime_parts) if (sup_parts or regime_parts) else "수급 데이터 없음"

    return warning, line1, line2


def get_signal_reason(score: float, rank: int, total: int,
                      macro_info: dict | None = None) -> str:
    """점수 기반 신호 이유 생성."""
    reasons = []

    # 점수 크기 해석
    if abs(score) > 0.8:
        strength = "강한"
    elif abs(score) > 0.4:
        strength = "보통"
    else:
        strength = "약한"

    if score > 0:
        reasons.append(f"{strength} 매수 신호")
        # 상위 몇 % 인지
        pct = rank / total * 100
        if pct <= 5:
            reasons.append("시장 상위 5%")
        elif pct <= 10:
            reasons.append("시장 상위 10%")
    else:
        reasons.append(f"{strength} 매도 신호")

    # 레짐 보너스
    if macro_info:
        kospi_regime = macro_info.get("kospi_regime", 0)
        rate_regime  = macro_info.get("rate_regime", 0)
        if kospi_regime > 0.5 and score > 0:
            reasons.append("강세장 모멘텀↑")
        if rate_regime > 0.5 and score > 0:
            reasons.append("금리하락 수혜")

    return " · ".join(reasons)


def fmt_signal(df: pd.DataFrame, top_n: int, market: str, horizon: int,
               macro_info: dict | None = None) -> str:
    date  = df["signal_date"].iloc[0]
    longs = df[df["direction"]=="LONG"].head(top_n)
    shorts= df[df["direction"]=="SHORT"].tail(top_n)
    total = len(df)

    lines = [
        f"📊 <b>{market} 매매 신호</b>",
        f"📅 {date}  |  예측기간: {horizon}거래일",
        f"",
    ]

    # 레짐 정보
    if macro_info:
        rate_regime  = macro_info.get("rate_regime", 0)
        kospi_regime = macro_info.get("kospi_regime", 0)
        rate_label   = "📉 금리하락" if rate_regime > 0.5 else "📈 금리상승"
        kospi_label  = "🐂 강세장" if kospi_regime > 0.5 else "🐻 약세장"
        kospi_mom    = macro_info.get("kospi_mom60", 0)
        rate_chg     = macro_info.get("rate10_chg20", 0)
        spread       = macro_info.get("yield_spread", 0)

        # 현재 레짐 해석
        if kospi_regime > 0.5 and rate_regime > 0.5:
            regime_msg = "모멘텀+금리 최적 구간 → 신호 신뢰도 ↑"
        elif kospi_regime > 0.5 and rate_regime <= 0.5:
            regime_msg = "강세장이나 금리 상승 주의 → 모멘텀 신호 중심"
        elif kospi_regime <= 0.5 and rate_regime > 0.5:
            regime_msg = "약세장+금리하락 → 방어주/배당주 유리"
        else:
            regime_msg = "약세장+금리상승 → 신호 신뢰도 낮음, 소규모 운용"

        regime_id = macro_info.get("regime_id", "A")
        ic_table  = REGIME_IC_BEST.get(regime_id, {})
        opt_h     = REGIME_HORIZON.get(regime_id, 20)
        lines += [
            f"📡 <b>현재 시장 레짐</b>",
            f"  {kospi_label} | {rate_label}",
            f"  KOSPI 60일: {kospi_mom:+.1%} | 금리변화: {rate_chg:+.2%}",
            f"  💡 {regime_msg}",
            f"",
            f"  IC by horizon:",
            f"  1주(fwd5)  {ic_table.get('fwd5',0):+.3f}  "
            f"1달(fwd20) {ic_table.get('fwd20',0):+.3f}  "
            f"3달(fwd60) {ic_table.get('fwd60',0):+.3f}",
            f"  📌 현재 기준: <b>fwd{opt_h}</b> 사용",
            f"",
        ]

    # 매수 종목 — 핵심 피처값 기반 이유
    ticker_metrics = get_ticker_metrics(longs["ticker"].tolist())
    # 레짐 IC 표시
    if macro_info:
        regime = macro_info.get("regime_id", "")
        if regime and regime in REGIME_IC:
            lines.append(
                f"  예측 IC: <b>{REGIME_IC[regime]:.4f}</b> "
                f"(레짐 {regime} 역사적 평균)"
            )
            lines.append("")
    lines.append(f"📈 <b>매수 상위 {top_n}종목</b>")
    for _, r in longs.iterrows():
        warning, line1, line2 = fmt_reason(
            str(r["ticker"]).zfill(6), r["score"], int(r["rank"]),
            total, ticker_metrics, macro_info
        )
        warn_icon = " ⚠️" if warning else ""
        # 3개 호라이즌 점수
        h_scores = ""
        for h, label in [(5,"1주"),(20,"1달"),(60,"3달")]:
            col = f"score_fwd{h}"
            if col in r.index and not pd.isna(r[col]):
                v = r[col]
                icon = "🟢" if v > 0 else "🔴"
                h_scores += f"{label}{icon}{v:+.2f} "

        lines.append(
            f"  {int(r['rank']):>2}. <b>{r['name'][:8]}</b> "
            f"<code>{r['score']:+.3f}</code>{warn_icon}"
        )
        if h_scores:
            lines.append(f"      {h_scores.strip()}")
        lines.append(f"      {line1}")
        lines.append(f"      {line2}")

    # 매도 종목 (이유 없이 간략히)
    lines += ["", f"📉 <b>매도 하위 {top_n}종목</b>"]
    for _, r in shorts.sort_values("rank").iterrows():
        lines.append(
            f"  {int(r['rank']):>3}. {r['name'][:8]} {str(r['ticker']).zfill(6)}"
            f"  <code>{r['score']:+.3f}</code>"
        )

    return "\n".join(lines)


def fmt_position(pos_df: pd.DataFrame, capital: float) -> str:
    total  = pos_df["actual_krw"].sum()
    remain = capital - total
    lines  = [
        f"💼 <b>포지션 ({len(pos_df)}종목)</b>",
        f"자본 {capital:,.0f}원  →  집행 {total:,.0f}원 ({total/capital:.0%})",
        f"현금 잔여 {remain:,.0f}원",
        "",
    ]
    body = ["<pre>티커     종목명       비중  주수    금액"]
    for _, r in pos_df.iterrows():
        body.append(
            f"{r['ticker']:<8}{r['name'][:8]:<9}"
            f"{r['weight']:>4.0%} {r['shares']:>4} {r['actual_krw']:>10,.0f}"
        )
    body.append("</pre>")
    return "\n".join(lines) + "\n".join(body)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def find_model(market: str, horizon: int, model_name: str | None) -> Path:
    """모델 파일 자동 탐색 — ensemble3 우선."""
    if model_name:
        p = MODEL_DIR / f"{model_name}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"모델 없음: {p}")
        return p

    # 우선순위: ensemble3 > best_ > ridge_ohlcv
    candidates = [
        MODEL_DIR / f"regime_v7_{market.lower()}_fwd{horizon}.pkl",
        MODEL_DIR / f"ensemble3_v6_{market.lower()}_fwd{horizon}.pkl",
        MODEL_DIR / f"ensemble3_v5_{market.lower()}_fwd{horizon}.pkl",
        MODEL_DIR / f"ensemble3_v3_{market.lower()}_fwd{horizon}.pkl",
        MODEL_DIR / f"ensemble3_{market.lower()}_fwd{horizon}.pkl",
        MODEL_DIR / f"best_ridge_{market.lower()}_ohlcv_fwd{horizon}.pkl",
        MODEL_DIR / f"ridge_{market.lower()}_ohlcv_fwd{horizon}.pkl",
    ]
    for p in candidates:
        if p.exists():
            logging.getLogger(__name__).info(f"  모델: {p.name}")
            return p
    raise FileNotFoundError(
        f"모델 없음. 먼저 colab_ensemble_v2.py 또는 07_save_model.py 실행 필요.\n"
        f"탐색 경로: {[str(c) for c in candidates]}"
    )


def main():
    parser = argparse.ArgumentParser(description="텔레그램 매매 신호 발송")
    parser.add_argument("--market",      choices=["KOSPI","KOSDAQ"], default="KOSPI")
    parser.add_argument("--horizon",     type=int, default=20)
    parser.add_argument("--topn",        type=int, default=30)
    parser.add_argument("--capital",     type=float, default=2_000_000)
    parser.add_argument("--max-price",   type=float, default=200_000)
    parser.add_argument("--model-name",  type=str, default=None,
                        help="모델 파일명 (확장자 제외). 없으면 ensemble3 자동 선택")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=2.0,
                        help="이 값 초과 |score| 종목 제거 (기본 2.0)")
    parser.add_argument("--universe", type=str, default=None,
                        help="유니버스 파일명 (kospi200 → universe_kospi200.csv)")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)
    load_env(override=True)

    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("  ⚠️  TELEGRAM_TOKEN/CHAT_ID 없음 — 미리보기 모드")

    # 유니버스
    if args.universe:
        univ_path = ROOT/"results"/"diagnose"/f"universe_{args.universe}.csv"
    else:
        univ_path = ROOT/"results"/"diagnose"/f"universe_filter_{args.market.lower()}.csv"
    universe  = pd.read_csv(univ_path)["ticker"].astype(str).str.zfill(6).tolist() if univ_path.exists() else None
    log.info(f"  유니버스: {len(universe) if universe else 0}종목 ({univ_path.name})")
    log.info(f"  유니버스: {len(universe) if universe else '전체'} 종목")

    # 패널
    panel = load_panel(args.market, args.no_download)

    # 모델
    model_path = find_model(args.market, args.horizon, args.model_name)

    # 신호
    signal_df = generate_signal(
        model_path, panel, universe,
        args.market, args.horizon, args.topn,
    )

    # 이상치 필터링
    signal_df = filter_outliers(signal_df, args.score_threshold, args.topn)

    # 저장
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = str(signal_df["signal_date"].iloc[0]).replace("-","")
    sig_path = SIGNALS_DIR / f"signal_{args.market.lower()}_{today}_fwd{args.horizon}.csv"
    signal_df.to_csv(sig_path, index=False, encoding="utf-8-sig")

    # 포지션
    pos_df = calc_positions(signal_df, args.capital, args.market, args.max_price)

    # 매크로 레짐 정보 수집
    macro_info = None
    try:
        from features.macro import load_macro
        macro_dir = ROOT / "data" / "macro"
        if macro_dir.exists():
            mdf = load_macro(macro_dir)
            latest = mdf.iloc[-1]
            rate_r = float(latest.get("macro_rate_regime", 0))
            bull_r = float(latest.get("macro_kospi_regime", 0))
            rid = "A" if bull_r>0.5 and rate_r>0.5 else                   "B" if bull_r>0.5 else                   "C" if rate_r>0.5 else "D"
            macro_info = {
                "rate_regime":  rate_r,
                "kospi_regime": bull_r,
                "kospi_mom60":  float(latest.get("macro_kospi_mom60", 0)),
                "rate10_chg20": float(latest.get("macro_rate10_chg20", 0)),
                "yield_spread": float(latest.get("macro_yield_spread", 0)),
                "regime_id":    rid,
            }
    except Exception as e:
        logging.getLogger(__name__).warning(f"  레짐 정보 로드 실패: {e}")

    # 메시지
    msg1 = fmt_signal(signal_df, args.topn, args.market, args.horizon, macro_info)
    msg2 = fmt_position(pos_df, args.capital)

    if token and chat_id:
        send_long(token, chat_id, msg1)
        time.sleep(1)
        send_long(token, chat_id, msg2)
        log.info("  ✅ 텔레그램 발송 완료")
    else:
        log.info("\n" + "="*55 + "\n[미리보기]\n" + "="*55)
        log.info(msg1)
        log.info("\n" + msg2)

    log.info(f"\n  신호 저장: {sig_path}")


if __name__ == "__main__":
    main()