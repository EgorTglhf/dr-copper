# Dr.Copper — A Quantitative Analysis Pipeline for Copper Price (HG=F)

> *"Dr.Copper" is the nickname traders give the metal for its supposed ability to diagnose the health of the global economy.*

A full quantitative research pipeline built on COMEX copper futures (HG=F), covering factor analysis, volatility modelling, machine learning signal generation, systematic backtesting, and derivatives pricing — implemented in Python with production-quality structure.

---

## Table of Contents

- [Motivation](#motivation)
- [Project Architecture](#project-architecture)
- [Key Results](#key-results)
- [Pipeline Phases](#pipeline-phases)
  - [Phase 1 — Data & Factor Analysis](#phase-1--data--factor-analysis)
  - [Phase 2 — ARIMA Baseline](#phase-2--arima-baseline)
  - [Phase 3 — Volatility Modelling](#phase-3--volatility-modelling)
  - [Phase 4 — Machine Learning Signal](#phase-4--machine-learning-signal)
  - [Phase 5 — Backtesting](#phase-5--backtesting)
  - [Phase 6 — Simulation & Risk](#phase-6--simulation--risk)
- [Limitations & Further Work](#limitations--further-work)

---

## Motivation

Copper is a uniquely rich asset for quantitative research. Unlike equities, its price is driven by a legible set of macro forces — Chinese and USA economy activity, the US dollar and Chinese yuan, global risk appetite, and physical supply constraints. This makes it possible to build interpretable factor models where PCA components map onto real economic narratives, not just statistical abstractions.

The project has three goals: demonstrate a rigorous end-to-end quant pipeline, show that ML signals can be grounded in economic intuition rather than black-box feature engineering, and produce honest performance evaluation.

---

## Project Architecture

```
    Raw data (via OpenBB)
             │
             ▼
┌─────────────────────────────┐
│  Phase 1: Feature pipeline  │  PCA on macro features → 3 interpretable factors
│  + regime clustering        │  KMeans on PC scores → 4 market regimes
└────────────┬────────────────┘
             │ PC scores + regime labels
             ▼
┌─────────────────────────────┐
│  Phase 2: ARIMA baseline    │  ARIMA, different approaches
│                             │  → simple models failed residual diagnostics
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 3: Vol modelling     │  ARIMA on returns, GARCH(1,1) on residuals
│  ARIMA + GARCH              │  → conditional vol series σ̂_t as downstream feature
└────────────┬────────────────┘
             │ GARCH σ̂_t
             ▼
┌─────────────────────────────┐
│  Phase 4: ML models         │  XGBoost + LSTM → ensemble signal
│  Walk-forward validation    │  Target: 5-day forward return sign {-1, 1}
└────────────┬────────────────┘
             │ copper_signal.parquet (1-day lagged)
             ▼
┌─────────────────────────────┐
│  Phase 5: Backtesting       │  Backtrader strategy + quantstats tearsheet
│  + performance analysis     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 6: Simulation & risk │  GBM + Heston MC → VaR/CVaR
│  QuantLib pricing           │  B-S option pricing, SABR smile calibration
└─────────────────────────────┘
```

---

## Key Results

### Backtesting performance (Phase 5)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| CAGR | `31.07%` | `13.52%` |
| Sharpe ratio | `0.84` | `0.59` |
| Sortino ratio | `1.09` | — |
| Calmar ratio | `0.44` | `0.54` |
| Max drawdown | `–70.93%` | `–25.27%` |
| Win rate | `53.11%` | — |
| Profit factor | `1.23` | — |

> The strategy generates significantly higher returns but carries substantially larger drawdown, driven primarily by the July 2025 tariff shock. — [analysed in Phase 5](#phase-5--backtesting)

> The combined strategy (MA cross + ML signals) results are reported in Key Results.

### Model comparison (Phase 4)

| Model | Accuracy score | ROC AUC score |
|---|---|---|
| LSTM | `55.20%` | `52.26%` |
| XGBoost | `55.10%` | `53.39%` |
| Ensemble | `60.00%` | `56.18%` |

### Risk metrics — Monte Carlo (Phase 6)

| Model | VaR 95% | VaR 99% | CVaR 95% | CVaR 99% |
|---|---|---|---|---|
| GBM | `22.71%` | `30.34%` | `27.42%` | `33.84%` |
| Heston | `23.18%` | `34.87%` | `30.09%` | `39.70%` |

> Heston produces approximately `17.3%` wider tails than GBM at the 99% CVaR level, reflecting the cost of assuming constant volatility.

### Volatility modelling (Phase 3)

> Persistence of `0.98` in the GARCH(1, 1) indicates long-memory volatility — typical for industrial metals and consistent with the literature.

---

## Pipeline Phases

### Phase 1 — Data & Factor Analysis

**Notebook:** [notebooks/Phase_1_Data_pipeline_PCA.ipynb](notebooks/Phase_1_Data_pipeline_PCA.ipynb)

**Data sources:**

| Series | Ticker / Code |
|---|---|
| Copper futures | HG=F |
| Crude oil futures | CL=F |
| Gold futures | GC=F |
| US Dollar Index | DX-Y.NYB |
| S&P 500 | ^SPX |
| US 10Y real yield | DFII10 |
| China CLI Index | — |
| USA  CLI Index | — |
| USDCNY | USDCNY |

**Methodology:**

Features are standardised and passed through PCA fitted only on the training data. The six retained components explain `65.9%` of total variance.
The top three of them explain `44.5%` of total variance and map onto interpretable economic factors:

| Component | Variance explained | Economic interpretation |
|---|---|---|
| PC1 | `20.4%` | Risk-on factor (commodities, S&P 500, USD loading) |
| PC2 | `13.7%` | Global growth (China/USA CLI, USD/CNY loading) |
| PC3 | `10.4%` | Defensive factor (gold, crude, SP500, DXY loading) |

KMeans clustering (k=4) on PC scores identifies four distinct market regimes:

| Cluster | Interpretation |
|---|---|
| 0 | `Risk-off regime or macro stress` |
| 1 | `Recovery` |
| 2 | `Monetary easing` |
| 3 | `Global growth with gold outperformance` |

---

### Phase 2 — ARIMA Baseline

**Notebook:** [notebooks/Phase_2_ARIMA_baseline.ipynb](notebooks/Phase_2_ARIMA_baseline.ipynb)

**Methodology:**

4 ARIMA models were implemented:
 - Auto ARIMA on daily log returns
 - Auto ARIMA on monthly log returns
 - ARIMA on monthly log returns with order chosen with ACF/PACF graphs
    - p = 12
    - d = 0
    - q = [2, 12, 14]
 - ARIMA on monthly raw prices with order chosen with ACF/PACF graphs
    - p = [2, 6, 9, 11, 16]
    - d = 1
    - q = 10

**Key finding:** `None of the models show success in residual diagnostics: the models either don't capture serial correlation or don't retain ARCH effects. Demands a deeper approach when moving to a GARCH model.`

---

### Phase 3 — Volatility Modelling

**Notebook:** [notebooks/Phase_3_GARCH_volatility_modelling.ipynb](notebooks/Phase_3_GARCH_volatility_modelling.ipynb)

**Methodology:**

After Phase 2, consider ARIMA on daily log returns with order selected via ACF/PACF graphs. These lags correspond to significant spikes in the ACF/PACF plots that exceed the confidence interval bounds and are distributed across multiple time horizons:
 - p = [5, 38, 101, 186]
 - d = 0
 - q = [5, 38, 101, 186]

Ljung-Box test on residuals confirms no remaining autocorrelation.
Jarque-Bera test indicates a Student's t-distribution.
Engle's ARCH LM test indicates the presence of ARCH effects.

The ARIMA model led to a constant log return, which is close to the historical mean return. So the GARCH model uses a constant mean.
The GARCH conditional volatility series σ̂_t is saved to `data/processed/features_with_vol.parquet` joined to other features and separately to `data/processed/garch_wf.parquet` and consumed by two downstream phases: as a feature in the Phase 4 ML models, and as the diffusion parameter σ in the Phase 6 GBM simulation.

**Key finding:** `Persistence of 0.98 in the GARCH(1, 1) indicates long-memory volatility — typical for industrial metals and consistent with the literature.`

---

### Phase 4 — Machine Learning Signal

**Notebook:** [notebooks/Phase_4_XGB_LSTM.ipynb](notebooks/Phase_4_XGB_LSTM.ipynb)

**Feature vector (per observation):**

```
crude_ret1d, crude_ret5d               ← Crude futures 1-day and 5-day log returns
gold_ret1d, gold_ret5d                 ← Gold futures 1-day and 5-day log returns
copper_ret1d, copper_ret5d             ← Copper futures 1-day and 5-day log returns
dxy_ret1d, dxy_ret5d                   ← DXY index 1-day and 5-day log returns
sp500_ret1d, sp500_ret5d               ← S&P500 index 1-day and 5-day log returns
copper_vol21d                          ← Annualized rolling volatility of daily log returns for copper prices over a 21-day window
cli_china, cli_usa                     ← Composite Leading Indicator (CLI) China and USA
DFII10                                 ← Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
USDCNY                                 ← US dollar / Chinese yuan exchange rate
garch_cond_vol                         ← conditional volatility
garch_cond_var                         ← conditional variance
garch_vol_ann                          ← annualized conditional volatility
persistence                            ← alpha + beta GARCH model
PCn                                    ← Principal Components (6 components)
regime_n                               ← one-hot encoded regime (4 classes)
```

**Target:** sign of 5-day forward copper return ∈ {−1, 1}.

**No look-ahead bias:** PCA and KMeans model trained only on the train dataset, no re-fitting on test data. Walk-forward approach for GARCH re-fitting.

**LSTM architecture:** input shape `(180, 29)` — sliding window of 180 days.

`Input(32, 180, 29) → LSTM(64) → LSTM(64) → LayerNorm(64) → Dropout(0.4) → Linear(64, 1) → Sigmoid()`

Training:
 - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
 - Loss: BCELoss
 - Max epochs: 100
 - Batch size: 32
 - Early stopping: patience=30

**XGBoost config:** `XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.15, subsample=0.7, colsample_bytree=0.5, reg_lambda=10, reg_alpha=5, min_child_weight=10)`

**Ensemble feature fusion approach:** Inspired by hybrid model approaches in the literature, we combine both by appending XGBoost predicted probabilities as an additional feature to the LSTM input.

**Key finding:** `LSTM captures temporal dependencies through sequential context, while XGBoost captures non-linear relationships in tabular macro features. The feature fusion approach allows the sequential model to condition on the tree-based signal at each timestep.`

---

### Phase 5 — Backtesting

**Notebook:** [notebooks/Phase_5_Backtesting.ipynb](notebooks/Phase_5_Backtesting.ipynb)

**Signal artifact:** `data/processed/copper_signal.parquet` — dated Series with values ∈ {−1, 1}, shifted 1 day to reflect realistic execution lag.

**Assumptions:**

| Parameter | Value |
|---|---|
| Transaction cost | `2.5 bps` per trade |
| Slippage | 0.0015 |
| Execution | Next-day open |
| Position sizing | Fixed % of portfolio |
| Benchmark | Buy & hold HG=F |

**Strategy:** The strategies tested were a simple MA cross, ML signal following and their combination.

**Full performance tearsheet:** [reports/Phase_5_tearsheet.html](reports/Phase_5_tearsheet.html)

**Model failure analysis — `July 2025, copper tariff shock`:**

`As observed in the tearsheet, July 2025 is the worst month for the strategy, with returns of -66.41%. The strategy is a combination of the simplest MA cross strategy and following ML signal, which makes it really vulnerable to black swans, like USA tariff shock. Probably, risk restrictions and news sentiment analysis can improve the situation. In other periods, the worst month shows only a -14.43% return.`

---

### Phase 6 — Simulation & Risk

**Notebook:** [notebooks/Phase_6_Monte_Carlo_Risk_and_Derivatives.ipynb](notebooks/Phase_6_Monte_Carlo_Risk_and_Derivatives.ipynb)

**Monte Carlo — GBM vs Heston:**

Both models simulate 10,000 price paths over 63 trading days. GBM uses last GARCH σ̂ as constant diffusion. Heston uses the same σ̂ as v0 with parameters κ=`4.3108`, θ=`0.0525` (long-run mean of GARCH series), ξ=`0.62`, ρ=`–0.65`. In production, κ, ξ, ρ would be calibrated to a live options vol surface.

**QuantLib option pricing:**

European call and put priced via Black-Scholes engine. Spot = `$6.3420`, σ = last GARCH σ̂ = `29.5544%`, T = 63 days, r = `0.0%`. Zero risk-free rate is assumed for simplification.

| | Call | Put |
|---|---|---|
| Strike | `$6.3420` | `$6.3420` |
| NPV | `$0.310462` | `$0.310462` |
| Delta | `0.524477` | `–0.475523` |
| Gamma | `0.511352` | `0.511352` |
| Vega | `1.049160` | `1.049160` |
| Theta | `-0.898226` | `-0.898226` |

**SABR calibration:** fitted to synthetic smile (strikes ±`20%` around spot). Parameters: α=`0.7217`, β=`0.50`, ρ=`-0.0216`, ν=`1.1995`. The SABR model captures the shape of the synthetic volatility smile very well.

**Key finding:** `Vol-of-vol in the Heston model and negative price/vol correlation fatten the left tail more than a fixed sigma does in GBM, which reflects a more realistic picture of the market.`

---

## Limitations & Further Work

**Data constraints.** LME warehouse stocks and full futures curve data (multi-tenor) require paid data providers. With Bloomberg or Refinitiv access, roll yield and curve shape would be natural additions to the feature set.

**Model limitations.** The current work doesn't aim for optimal performance of the used ML models. The models are trained without hyperparameter tuning. Transformer-based architectures would be a natural next step.

**Heston calibration.** Heston parameters are set to reasonable starting values for copper rather than calibrated to a real vol surface. Calibration to COMEX copper options data would be the correct next step for the simulation section.

**Further work:**
- Extend feature set with shipping rates (BDI), LME inventory and PMI index
- Online learning / rolling refit to reduce regime sensitivity
- Calibrate Heston to real copper options vol surface

---

*Built as a quantitative finance portfolio project. All results are out-of-sample on walk-forward test folds. Past performance of any simulated strategy does not imply future results.*
