# Multi-Asset RL Trading for the Indian Market

This project is a research prototype for multi-asset portfolio allocation across
Indian equities, mutual-fund NAVs, and gold ETF proxies. It includes a custom
Gymnasium environment, RSI/MACD/SMA and macro features, recurrent DQN and
Double-DQN agents, a continuous-allocation PPO agent, and optional FinBERT
sentiment features. PPO held-out results are reported beside cash and
equal-weight buy-and-hold baselines; no agent is presumed to outperform them.

## Install

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[data,deep-learning,test]"
```

Add the `sentiment` extra when using news inference:

```powershell
python -m pip install -e ".[sentiment]"
```

## Choose your investable universe

The agent does **not** discover every investment available in India. It
optimizes only the assets in the price/NAV CSV you provide. Replace or extend
[`data/universe.example.csv`](data/universe.example.csv) to choose the assets:

```csv
asset_class,ticker,scheme_code,label
stock,RELIANCE.NS,,
gold_etf,GOLDBEES.NS,,
mutual_fund,,118347,IDBI_NIFTY50_DIRECT
```

Use Yahoo Finance tickers for `stock`, `etf`, and `gold_etf`; use an MFAPI
scheme code for `mutual_fund`. The example universe spans several equity
sectors, a broad-market ETF, a gold ETF, and two Nifty 50 index funds. Replace
it with the stocks, funds, ETFs, or other supported tickers you want to
compare. Index membership changes over time, and the example is not a
recommendation.

## Fetch data

Install the Yahoo Finance downloader and fetch adjusted equity/gold-ETF prices,
a mutual-fund NAV series, India VIX/rupee/crude market proxies, and the latest
Yahoo Finance RSS headlines for your selected universe:

```powershell
python -m pip install -e ".[data]"
finance-rl-fetch --universe-csv data/universe.example.csv --start 2014-10-04 --end 2024-10-04 --output data/raw
```

The command writes aligned `prices.csv`, `macro.csv`, `news.csv`, and
`metadata.json` under `data/raw/`. You can instead pass tickers directly with
`--symbols ...` and scheme codes with `--funds ...`. A common date range is used
for all selected assets; new funds/ETFs with later inception dates will shorten
the usable history. `GOLDBEES.NS` is a gold ETF proxy, not physical gold. Macro
columns are market proxies, not a complete official RBI dataset. RSS provides
current headlines, not historical news, and must not be treated as historical
training data. FinBERT needs the optional `sentiment` extra and downloads its
model on first use.

## Run training

Train continuous PPO allocations across all assets in a price file:

```powershell
finance-rl --prices-csv data/raw/prices.csv --macro-csv data/raw/macro.csv --agent ppo
```

The agent allocates across each CSV column plus cash. Choose `--agent dqn` for
standard DQN or `--agent ddqn` for DDQN-LSTM, which use discrete per-asset
buy/sell/hold actions. Training uses the first 80% of rows in chronological
order; the remaining rows are held out for deterministic evaluation. Change
the cutoff with `--train-fraction`.

The evaluation JSON saves final weights by asset as well as cash/equity. Results
are also compared with cash and equal-weight buy-and-hold baselines.

`macro.csv` should have a date index and numeric macro columns. To score recent
headlines, pass `--news-csv data/raw/news.csv`; FinBERT uses its
`published_at` and `text` fields. Macro and sentiment features are shifted one
trading row by the feature builder to reduce look-ahead leakage. Macro
publication timing should still be checked for the chosen source.

## Derivatives

Futures and options are **not currently modeled as tradable derivatives**.
Passing a futures or option premium as an ordinary price column would omit
contract multipliers, margin, leverage, expiry/roll, settlement, and (for
options) strike and volatility effects. A valid derivatives extension needs
historical contract-level data and these mechanics implemented in the
environment; the current results only cover cash assets and ETF/NAV proxies.

Downloaded raw datasets are excluded from Git by `.gitignore`; users fetch
their chosen assets locally. Models, episode rewards, and held-out evaluation metrics are saved under
`artifacts/` by default. Training is for experimentation only and does not constitute investment advice. Historical
data can contain survivorship, corporate-action, availability, and execution
biases; always evaluate on a strictly held-out period and include realistic
trading costs before interpreting results.

## Environment and action space

The observation is a `(lookback, features)` array containing each asset's
return, normalized RSI, MACD and SMA gap; optional lagged macro/sentiment
features; and current cash/asset weights. The discrete actions are hold, buy a
fixed fraction of portfolio value in one asset, or sell that fraction of one
asset. Short positions are not allowed. Rewards are log changes in portfolio
equity after transaction costs.

## Tests

```powershell
python -m pip install -e ".[test]"
pytest
```
