# 🐋 Ride the Whale — KOSPI Smart Money Alpha System

> A quantitative trading system for the Korean KOSPI market that extracts alpha by tracking foreign and institutional investor (smart money) supply/demand flows, with regime-adaptive models.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Core Idea

**"Don't watch where the whales have been. Watch where they're starting to move."**

Foreign investors ("whales") in KOSPI have a proven information advantage (Choe, Kho & Stulz 1999, 2005). But the signal is counterintuitive:

```
20-day accumulated foreign buying  → CONTRARIAN signal (t = -7.09)
Foreign buying ACCELERATION        → MOMENTUM signal  (t = +3.26)
```

Stocks where foreigners have been buying for 20 days have already moved. The alpha lies in catching the *moment they start*.

---

## 📐 Mathematical Foundations

### 1. Fama-MacBeth Cross-Sectional Regression (1973)

Feature selection uses daily cross-sectional regressions:

$$\beta_t = \frac{1}{N} \sum_{i=1}^{N} \text{rank}(X_{i,t}) \cdot \text{rank}(R_{i,t+20})$$

$$\bar{\beta} = \frac{1}{T} \sum_{t=1}^{T} \beta_t, \quad t\text{-stat} = \frac{\bar{\beta}}{\text{std}(\beta_t) / \sqrt{T}}$$

**Decision rule:** $|t| > 2$ → feature is statistically significant.

**Key FM results (2018–2024, N=270 stocks, T=1,721 days):**

| Feature | Mean IC | t-stat | Direction |
|---|---|---|---|
| `tech_52w_pos` | +0.021 | +14.40 | ✅ Momentum |
| `tech_vol20` | -0.073 | -19.41 | ✅ Low vol wins |
| `sup_fmom_chg` | +0.008 | +3.26 | ✅ Foreign accel |
| `sup_fcum20` | -0.014 | -7.09 | ✅ Contrarian |
| `earn_accel` | +0.011 | +6.46 | ✅ Earnings acceleration |
| `earn_rev_yoy` | -0.017 | -7.75 | ✅ Contrarian |
| `short_ratio` | -0.006 | -3.02 | ✅ Short pressure |
| `earn_qoq` | -0.001 | -0.54 | ❌ Noise → removed |
| `short_squeeze` | -0.003 | -1.09 | ❌ Noise → removed |

---

### 2. Information Coefficient (IC)

**IC = Spearman rank correlation between predicted scores and actual returns:**

$$IC_t = \rho_s(\hat{y}_{i,t},\ y_{i,t+20})$$

$$\bar{IC} = \frac{1}{T}\sum_{t=1}^T IC_t, \quad Z = \frac{\bar{IC}}{\sigma_{IC}/\sqrt{T}}$$

**Interpretation:**

| IC | Meaning |
|---|---|
| < 0.02 | No signal |
| 0.02–0.05 | Weak alpha (tradeable) |
| **0.05–0.10** | **Strong alpha (hedge fund grade)** |
| > 0.10 | Exceptional (verify for overfitting) |

**Grinold's Fundamental Law:**
$$IR = IC \times \sqrt{Breadth}$$

With 270 stocks × 252 days = 68,040 bets/year, even IC=0.04 generates IR≈0.65.

---

### 3. Foreign Investor Information Advantage

**Theoretical basis: Kyle (1985) price impact model**

$$\Delta p = \lambda \cdot (Q_f - Q_r)$$

Where $Q_f$ = informed (foreign) order flow, $Q_r$ = retail order flow, $\lambda$ = market depth.

**Empirical validation for Korea:**
- Choe, Kho & Stulz (1999, JFE): Foreign herding precedes price increases
- Choe, Kho & Stulz (2005, RFS): Foreigners trade at better prices than locals
- **Our finding**: The *acceleration* of foreign buying ($\Delta Q_f / \Delta t$) is more predictive than the level

---

### 4. Factor Momentum & Earnings Acceleration

**Ehsani & Linnainmaa (2022, JF) — Factor Momentum:**

Past factor returns predict future factor returns. Applied to earnings:

$$\text{earn\_accel}_t = \frac{\Delta\text{OI}_t/\text{OI}_{t-1} - \Delta\text{OI}_{t-1}/\text{OI}_{t-2}}{1}$$

This is the **second derivative** of operating income. Stocks with accelerating earnings (not just high earnings) have the strongest forward returns.

**Why earnings LEVEL is contrarian (earn_yoy t=-7.12):**
Stocks that already had high earnings growth have already been priced in → mean reversion.

**Why earnings ACCELERATION is momentum (earn_accel t=+6.46):**
Acceleration signals a structural improvement not yet fully priced.

---

### 5. Regime-Based Model Theory

Market regimes follow a hidden Markov process:

$$s_t \in \{A, B, C, D\}$$

$$P(s_{t+1} | s_t) = \text{transition matrix}$$

We approximate regime using observable macro variables:

$$s_t = f(\text{KOSPI}_{200MA},\ \Delta\text{rate}_{20d})$$

| Regime | KOSPI | Rate | IC | Features |
|---|---|---|---|---|
| A | Bull | Falling | **0.097** | 76 (momentum) |
| B | Bull | Rising | 0.062 | 76 (momentum) |
| C | Bear | Falling | 0.011 | 80 (fundamental) |
| D | Bear | Rising | 0.054 | 80 (fundamental) |

**Why feature sets differ by regime (Ehsani & Linnainmaa 2022):**
- Bull markets: Momentum/supply signals dominate
- Bear markets: Fundamental signals (earnings, short pressure) become more informative

---

### 6. Short Selling as Signal

**Short ratio as contrarian indicator:**

$$\text{short\_ratio}_t = \frac{\text{short\_volume}_t}{\text{total\_volume}_t}$$

High short ratio → strong selling pressure → negative expected return (t=-3.02).

**Short squeeze (removed — t=-1.09):**
The squeeze signal (high short ratio + rising price) was theoretically appealing but statistically insignificant in our universe, likely because KOSPI short covering dynamics differ from US markets.

---

### 7. Z-test for Model Validity

Ongoing model monitoring uses a rolling Z-test:

$$Z = \frac{\bar{IC}}{\sigma_{IC}/\sqrt{T}}$$

| Z | Interpretation |
|---|---|
| > 2.0 | 95% confidence signal is real |
| > 3.0 | 99% confidence |
| < 2.0 | May be noise, monitor |

**Note (Zhang et al. 2020):** Detecting IC deterioration of 0.01 requires 12+ months of data. 3-month results are indicative only.

---

## 🏗️ Model Architecture (v7)

```
Daily Data (OHLCV + Supply/Demand + Short + Earnings)
    ↓
Macro Regime Detection
  KOSPI 200-day MA → Bull/Bear
  Bond ETF 20-day change → Rate Falling/Rising
    ↓
Regime A/B: 76-Feature Momentum Model
  Ridge (α=0.01) × 0.5 + ElasticNet × 0.5
    ↓
Regime C/D: 80-Feature Fundamental Model
  Ridge (α=0.01) × 0.5 + ElasticNet × 0.5
    ↓
Signal Score → Top 20 BUY / Bottom 20 SELL
```

---

## 📊 Feature Set

### Core Features (76, used in all regimes)

**Technical (11):** `tech_52w_pos`★, `tech_bb_pos`, `tech_tv_ratio`, `tech_macd`, `tech_mom20`, `tech_vol20`, `tech_mom60`, `tech_vol_ratio`, `tech_hl_spread`, `tech_ma20_dev`, `tech_ma5_dev`

**Supply/Demand (9):** `sup_fmom_chg`★, `sup_fmom_accel`, `sup_fnet_rank`, `sup_fcum10`, `sup_istreak`, `sup_fcum20`, `sup_icum20`, `sup_fstreak`, `sup_divergence`

**Macro (6):** `macro_kospi_regime`, `macro_kospi_mom60`, `macro_rate10_chg20`, `macro_yield_spread`, `macro_spread_chg20`, `macro_rate_regime`

**Regime Interactions (20):** `ix_bull_tech_52w_pos`★, `ix_rate_tech_vol20`, ... *(bull/rate × top 10 features)*

**Derivatives/1st-diff (30):** `d5_macro_rate10_chg20`★, `d20_tech_mom60`, ... *(5d and 20d change of 15 features)*

★ = EN-selected core signals

### Fundamental Features (4, Regime C/D only)

| Feature | t-stat | Theory |
|---|---|---|
| `earn_accel` | +6.46 | Earnings 2nd derivative (Ehsani & Linnainmaa 2022) |
| `earn_rev_yoy` | -7.75 | Revenue YOY (contrarian) |
| `earn_yoy` | -3.35 | Op. income YOY (contrarian) |
| `short_ratio` | -3.02 | Short pressure (Kyle 1985) |

---

## 📈 Performance History

| Version | Val IC | Key Change |
|---|---|---|
| v2 | 0.017 | 35 raw features |
| v3 | 0.075 | Fama-MacBeth feature selection |
| v6 | 0.038 | Robust (val+test balanced) |
| **v7 Regime A** | **0.097** | Regime-adaptive models ← Current |
| v7 Regime C | 0.011 | Bear market w/ fundamentals |

**Training period:** Rolling 3-year window (2018–2024)

**Why v3 got 0.075 in 2024:** 2024 was predominantly Regime A (bull + rate falling). v7 explicitly models this, achieving IC=0.097 in that regime vs 0.038 for v6's single model.

---

## 🔧 Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Variables (`.env`)
```bash
KRX_ID=your_krx_id
KRX_PW=your_krx_password
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DART_API_KEY=your_dart_key     # dart.fss.or.kr (free)
LOG_LEVEL=INFO
```

### Initial Data Download
```bash
# 1. Full panel (7 years, ~2 hours)
python3 scripts/01_download.py --market KOSPI --workers 2

# 2. Universe filter (>2B KRW daily volume)
python3 scripts/02_diagnose.py --market KOSPI

# 3. Short selling data (server)
python3 scripts/local_fetch_short.py

# 4. Earnings data via DART API (server)
python3 scripts/fetch_earnings.py
```

### Train Model (Google Colab)
```python
os.chdir('/content/drive/MyDrive/project/smart_money')
!python3 scripts/colab_fama_macbeth_v2.py  # Feature validation first
!python3 scripts/colab_train_v7.py          # Regime-based training
```

### Deploy to Server
```bash
# Copy model
docker cp regime_v7_kospi_fwd20.pkl CONTAINER:/root/smart_money/models/saved/

# Start scheduler
nohup python3 scripts/11_scheduler.py > logs/scheduler.log 2>&1 &
```

---

## 📅 Daily Operation (Fully Automated)

```
16:00 KST  10_update_data.py      OHLCV + supply/demand + short selling
16:05 KST  (Monday only)          Macro ETF update (bond + KOSPI index)
16:10 KST  12_monitor.py          Stop-loss / take-profit alerts
           14_consecutive_signal.py  Stocks appearing 3+ days in a row
           13_ic_tracker.py       Actual IC validation (after 20 trading days)
16:30 KST  09_telegram.py         Regime detection → signal → Telegram
```

**Telegram message format:**
```
📊 KOSPI Signal | 2026-06-18

📡 Current Regime
  🐂 Bull Market | 📉 Rate Falling
  KOSPI 60d: +69.6% | Rate Δ: -1.14%
  Predicted IC: 0.0967 (Regime A historical avg)
  💡 Momentum + rate optimal → Signal confidence ↑

📈 Top 20 BUY
  1. 삼성전자 005930  +0.156
     📍52w 100% · 🔺20d +31.6%
     🐋FgnAccel +4.2% · 🟢Fgn10d +1.9%↑ · 🏦Inst5d consecutive↑

  ⚠️ = Foreign selling despite high rank
  📉페이딩 = Spike fading (high 10d cumul + recent decel)
```

---

## 🔄 Maintenance Schedule

### Monthly — Universe Update
```bash
python3 scripts/01_download.py --market KOSPI --workers 2
python3 scripts/02_diagnose.py --market KOSPI
wc -l results/diagnose/universe_filter_kospi.csv  # should be ~271
```

### Quarterly — Model Retraining (Mar/Jun/Sep/Dec)
```python
# 1. Update earnings data
!python3 scripts/fetch_earnings.py

# 2. Update training window in colab_train_v7.py
FULL_START = "2019-01-01"  # slide forward 1 year
FULL_END   = "2025-12-31"

# 3. Retrain
!python3 scripts/colab_train_v7.py

# 4. Check regime IC improvement
# If Regime A IC > 0.09 → deploy
# If Regime A IC < 0.06 → investigate
```

### Model Validity Check (from 13_ic_tracker.py)
```
IC > 0.03  → ✅ Valid, continue
IC 0.01–0.03 → ⚠️  Weakening, schedule retraining
IC < 0.01  → ❌ Retrain immediately
IC < 0     → 🚨 Stop trading
```

---

## ⚠️ Risk Management

| Rule | Threshold | Source |
|---|---|---|
| Stop-loss | -7% from signal date close | Telegram alert |
| Take-profit | +15% from signal date close | Telegram alert |
| Time exit | 20 trading days | Telegram alert |
| Short entry | Position enters bottom-20 | Telegram alert |
| Fading signal | `fcum10 > 20% AND fmom_chg < 0` | ⚠️ Warning label |
| Outlier filter | `\|score\| > 2.0` | Auto-excluded |
| High-price filter | Price > capital/n | Auto-excluded |

---

## 📁 Project Structure

```
ride-the-whale/
├── features/
│   ├── technical_v2.py         # FM-selected technical features (11)
│   ├── supply_v2.py            # FM-guided supply/demand features (9)
│   ├── macro.py                # Interest rate + regime features (6)
│   ├── derivatives.py          # 1st-derivative features (30)
│   ├── short_selling.py        # Short selling features (1 after FM)
│   ├── earnings.py             # Earnings event features (3 after FM)
│   └── pipeline_v2.py          # Feature pipeline
├── scripts/
│   ├── 01_download.py          # Full panel download
│   ├── 02_diagnose.py          # Universe filtering
│   ├── 09_telegram.py          # Signal generation + Telegram
│   ├── 10_update_data.py       # Daily data update (OHLCV + short)
│   ├── 11_scheduler.py         # Automated scheduler
│   ├── 12_monitor.py           # Stop-loss/take-profit monitoring
│   ├── 13_ic_tracker.py        # Actual IC auto-validation (H)
│   ├── 14_consecutive_signal.py # Consecutive signal tracking (B)
│   ├── fetch_earnings.py       # DART API earnings collection
│   ├── local_fetch_short.py    # Short selling data collection
│   ├── colab_train_v7.py       # Regime-based model training
│   └── colab_fama_macbeth_v2.py # FM feature validation (84 features)
├── labels/
│   └── forward_return.py       # fwd20 label generation
├── data/
│   └── macro/                  # Interest rate ETF data (versioned)
├── models/
│   └── saved/                  # Trained pkl files (gitignored)
├── results/
│   ├── diagnose/               # Universe files
│   ├── signals/                # Daily signal CSVs (gitignored)
│   └── ic_tracker/             # IC validation results
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📚 Academic References

| Paper | Application in this system |
|---|---|
| Fama & MacBeth (1973, JPE) | Cross-sectional feature selection via t-stat |
| Choe, Kho & Stulz (1999, JFE) | Foreign investor herding in Korea |
| Choe, Kho & Stulz (2005, RFS) | Foreign investor information advantage |
| Kyle (1985, Econometrica) | Price impact model → `sup_fnet_rank`, `short_ratio` |
| Jegadeesh & Titman (1993, JF) | Momentum → `tech_52w_pos`, `tech_mom20` |
| Amihud (2002, JFM) | Illiquidity → `tech_vol20` (reverse) |
| Ehsani & Linnainmaa (2022, JF) | Factor momentum → `earn_accel` (2nd derivative) |
| Grinold & Kahn (2000) | IC/IR framework, Fundamental Law |
| Zhang et al. (2020, arXiv) | Rolling Z-test for IC monitoring |

---

## ⚡ Key Parameters

```python
# Universe filter (scripts/02_diagnose.py)
MIN_TRADE_VALUE = 2e9        # 2B KRW daily avg

# Model (scripts/colab_train_v7.py)
FULL_START = "2018-01-01"    # Training window start
FULL_END   = "2024-12-31"    # Training window end
HORIZON    = 20              # Forward return days

# Regime thresholds (features/macro.py)
BULL_THRESHOLD = 0           # KOSPI above 200-day MA
RATE_THRESHOLD = 0           # Bond ETF 20-day return > 0

# Risk (scripts/12_monitor.py)
STOP_LOSS   = -0.07          # -7%
TAKE_PROFIT =  0.15          # +15%
HOLD_DAYS   = 20             # Max holding period

# Signal filter (scripts/09_telegram.py)
SCORE_THRESHOLD = 2.0        # |score| > 2.0 excluded
MAX_PRICE = 1_000_000        # High-price exclusion
FADING_CUMUL = 20.0          # fcum10 > 20% = fading warning
FADING_ACCEL = 0             # fmom_chg < 0 = fading warning
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

*"Don't fight the whales. Ride them."*
