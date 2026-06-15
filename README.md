# 🐋 Ride the Whale — KOSPI Smart Money Alpha System

> A quantitative trading system for the Korean KOSPI market that extracts alpha by tracking foreign and institutional investor (smart money) supply/demand flows.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

**Core Idea:** Foreign investors are "whales" in KOSPI. Instead of following their long-term accumulation (which is a contrarian signal), we track the *acceleration* of their buying — the moment they start moving.

**Key Findings (Fama-MacBeth, t-stat):**
```
sup_fmom_chg    t = +3.26  ← Foreign buying acceleration (core signal)
tech_52w_pos    t = +14.40 ← 52-week high position (strongest technical)
sup_fcum20      t = -7.09  ← 20-day accumulated buying = CONTRARIAN signal
sup_icum20      t = -8.19  ← 20-day institutional = also CONTRARIAN
```

**Insight:** "When foreigners have been buying for 20 days, the stock has already moved. What matters is when they *start* buying."

---

## 🏗️ Model Architecture

### Ensemble: `ridge_full × 0.5 + en_full × 0.5`

```
76 Features → Ridge Regression (α=0.01)  ─┐
             └→ Elastic Net (L1+L2)       ─┴─→ Weighted Average → Signal Score
```

### Feature Set (76 total)

| Category | Count | Key Features |
|---|---|---|
| Technical (v2) | 11 | `tech_52w_pos`, `tech_vol20`, `tech_mom60` |
| Supply/Demand (v2) | 9 | `sup_fmom_chg`, `sup_fcum10`, `sup_icum20` |
| Macro | 6 | `macro_kospi_regime`, `macro_rate10_chg20`, `macro_yield_spread` |
| Regime Interactions | 20 | `ix_bull_tech_52w_pos`, `ix_rate_sup_fmom_chg` |
| Derivatives (1st diff) | 30 | `d5_macro_rate10_chg20`, `d20_sup_fmom_chg` |

### EN Selected Features (most important)
```
ix_bull_tech_52w_pos     ← Momentum amplified in bull market (regime interaction)
d5_macro_rate10_chg20    ← Rate change acceleration (2nd derivative of rates)
sup_fmom_chg             ← Foreign buying velocity
```

---

## 📊 Performance

| Version | Val IC (2023) | Test IC (2024) | Notes |
|---|---|---|---|
| v2 Baseline | +0.017 | 0.038 | 35 raw features |
| v3 FM-filtered | +0.000 | 0.075 | Fama-MacBeth feature selection |
| v5 +Interactions | +0.017 | 0.052 | Regime × feature interactions |
| **v6 Final** | **+0.038** | **0.040** | +Derivative features, most robust |

**Training Period (Rolling 3-Year):**
```
Train: 2020-01-01 ~ 2022-12-31
Val:   2023-01-01 ~ 2023-12-31
Test:  2024-01-01 ~ 2024-12-31
```

**Optimal Regime (highest IC):**
```
✅ KOSPI bull market (above 200-day MA) + Rate falling = Best conditions
⚠️ Bear market + Rate rising = Lowest IC, reduce position size
```

---

## 🔧 Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Variables (`.env`)
```bash
# KRX Login (required for supply/demand data)
KRX_ID=your_krx_id
KRX_PW=your_krx_password

# Telegram Bot
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Initial Data Download
```bash
# 1. Download 7-year KOSPI panel (takes ~2 hours)
python3 scripts/01_download.py --market KOSPI --workers 2

# 2. Filter universe (stocks with avg daily volume > 2B KRW)
python3 scripts/02_diagnose.py --market KOSPI

# 3. Download macro data (run on local machine with KRX access)
python3 scripts/local_fetch_macro.py
```

### Train Model (Google Colab recommended)
```python
# Upload project to Google Drive, then:
import os
os.chdir('/content/drive/MyDrive/project/smart_money')
!python3 scripts/colab_train_v6.py
```

### Deploy to Server
```bash
# Copy trained model
docker cp ensemble3_v6_kospi_fwd20.pkl container_id:/root/smart_money/models/saved/

# Start scheduler
cd /root/smart_money
nohup python3 scripts/11_scheduler.py > logs/scheduler.log 2>&1 &
```

---

## 📅 Daily Operation (Automated)

```
16:00 KST  → 10_update_data.py   Download today's OHLCV + supply data
16:10 KST  → 12_monitor.py       Check stop-loss/take-profit alerts
16:30 KST  → 09_telegram.py      Generate signals + send Telegram
```

**Telegram Output:**
```
📊 KOSPI Signal | 2026-06-15
📡 Current Regime
  🐂 Bull Market | 📈 Rate Rising
  KOSPI 60d: +12.3% | Rate Δ: +0.15%
  💡 Bull + Rate rising → Focus on momentum

📈 Top 20 BUY
  1. SK이터닉스 475150  +1.339
     ↳ Strong buy · Top 5%
...
```

---

## 🔄 Maintenance Schedule

### Monthly — Universe Update
```bash
# Run on server (1st Monday of each month, ~30 min)
python3 scripts/01_download.py --market KOSPI --workers 2
python3 scripts/02_diagnose.py --market KOSPI

# Verify
wc -l results/diagnose/universe_filter_kospi.csv
# Should show ~270 stocks
```

### Quarterly — Model Retraining
```python
# 1. Update training periods in colab_train_v6.py
TRAIN_START = "2021-01-01"  # Always recent 3 years
TRAIN_END   = "2023-12-31"
VAL_START   = "2024-01-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"

# 2. Upload latest panel to Google Drive
# (server → local → Google Drive)

# 3. Run training
!python3 scripts/colab_train_v6.py

# 4. Check IC comparison table in output
# If new_IC > 0.03 on both val and test → deploy
# If new_IC < 0.01 → investigate data issues

# 5. Deploy new model
docker cp ensemble3_v6_kospi_fwd20.pkl container_id:/root/smart_money/models/saved/
```

### Macro Data Update
```bash
# Run locally with KRX access whenever macro data is >7 days old
python3 scripts/local_fetch_macro.py

# Upload to server:
docker cp data/macro/ container_id:/root/smart_money/data/
```

---

## 📈 IC Validation (After 3 Months)

```python
import pandas as pd, glob
from scipy.stats import spearmanr

# Load all signal CSVs
signals = pd.concat([pd.read_csv(f) for f in glob.glob('results/signals/*.csv')])

# Compare with actual returns after 20 trading days
# (requires price data for t+20)
actual_returns = ...  # load from panel

ic_daily = signals.groupby('signal_date').apply(
    lambda g: spearmanr(g['score'], actual_returns.loc[g.index]).statistic
)

print(f"Mean IC:  {ic_daily.mean():.4f}")
print(f"Z-stat:   {ic_daily.mean() / (ic_daily.std() / len(ic_daily)**0.5):.2f}")
print(f"IC > 0:   {(ic_daily > 0).mean():.1%}")
```

**Decision Rules:**
```
IC > 0.03  → ✅ Model valid, continue
IC 0.01~0.03 → ⚠️  Signal weakening, schedule retraining
IC < 0.01  → ❌ Retrain immediately
IC < 0     → 🚨 Stop trading, full review
```

---

## ⚠️ Risk Management (Built into Signals)

| Rule | Threshold | Action |
|---|---|---|
| Stop-loss | -7% from signal date close | Auto Telegram alert |
| Take-profit | +15% from signal date close | Auto Telegram alert |
| Time exit | 20 trading days | Auto Telegram alert |
| Short entry | Position appears in bottom 20 | Auto Telegram alert |
| Outlier filter | `\|score\| > 2.0` | Auto excluded from signals |
| High-price filter | Stock price > capital/n | Auto excluded |

---

## 📁 Project Structure

```
ride-the-whale/
├── features/
│   ├── technical_v2.py      # FM-selected technical features (11)
│   ├── supply_v2.py         # FM-guided supply features (9)
│   ├── macro.py             # Interest rate + regime features (6)
│   ├── derivatives.py       # 1st derivative features (30)
│   └── pipeline_v2.py       # Feature pipeline
├── scripts/
│   ├── 01_download.py       # Full panel download
│   ├── 02_diagnose.py       # Universe filtering
│   ├── 09_telegram.py       # Signal generation + Telegram
│   ├── 10_update_data.py    # Daily data update
│   ├── 11_scheduler.py      # Automated scheduler
│   ├── 12_monitor.py        # Position monitoring
│   ├── colab_train_v6.py    # Model training (Colab)
│   ├── colab_fama_macbeth.py # Feature validation
│   └── local_fetch_macro.py  # Macro data download
├── labels/
│   └── forward_return.py    # fwd20 label generation
├── data/
│   └── macro/               # Interest rate ETF data
├── models/
│   └── saved/               # Trained pkl files (gitignored)
├── results/
│   ├── diagnose/            # Universe files
│   └── signals/             # Daily signal CSVs (gitignored)
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📚 Academic References

| Paper | Relevance |
|---|---|
| Choe, Kho, Stulz (1999, JFE) | Foreign investor herding in Korea |
| Choe, Kho, Stulz (2005, RFS) | Foreign investor information advantage |
| Jegadeesh & Titman (1993, JF) | Momentum strategy |
| Ehsani & Linnainmaa (2022, JF) | Factor momentum → basis for derivative features |
| Kyle (1985, Econometrica) | Price impact model → `sup_fnet_norm` |
| Amihud (2002, JFM) | Illiquidity measure |
| Fama & MacBeth (1973, JPE) | Cross-sectional regression → feature selection |
| Grinold & Kahn (2000) | IC/IR framework |

---

## ⚡ Key Parameters

```python
# Universe filter (scripts/02_diagnose.py)
MIN_TRADE_VALUE = 2e9     # 2B KRW daily avg → change for broader/narrower universe

# Model (colab_train_v6.py)
TRAIN_YEARS = 3           # Rolling window → increase for more history
HORIZON     = 20          # Forward return days → match your holding period

# Risk (scripts/12_monitor.py)
STOP_LOSS   = -0.07       # -7% stop loss
TAKE_PROFIT =  0.15       # +15% take profit
HOLD_DAYS   = 20          # Max holding period

# Signal filter (scripts/09_telegram.py)
SCORE_THRESHOLD = 2.0     # |score| > 2.0 = data anomaly, excluded
MAX_PRICE = 1_000_000     # Stocks above this price excluded from position sizing
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

*"Don't fight the whales. Ride them."*
