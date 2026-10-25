"""Price, macroeconomic, and optional FinBERT feature preparation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sentiment: pd.Series | None = None,
) -> pd.DataFrame:
    """Build aligned, finite features without using future observations.

    ``prices`` must contain one positive close-price column per asset. Macro
    values and sentiment are lagged by one trading row before being exposed.
    """
    if prices.empty or prices.columns.empty:
        raise ValueError("prices must contain at least one asset and one row")
    if prices.columns.has_duplicates or prices.index.has_duplicates:
        raise ValueError("prices must have unique asset names and dates")
    closes = prices.astype(float).sort_index()
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any().any():
        raise ValueError("prices must contain only finite, positive values")

    features: dict[str, pd.Series] = {}
    for asset in closes.columns:
        close = closes[asset]
        delta = close.diff()
        gains = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=1, adjust=False).mean()
        losses = -delta.clip(upper=0).ewm(alpha=1 / 14, min_periods=1, adjust=False).mean()
        relative_strength = gains / losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + relative_strength))
        rsi = rsi.mask((losses == 0) & (gains > 0), 100).fillna(50)

        ema_fast = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        sma = close.rolling(window=20, min_periods=1).mean()

        features[f"{asset}__return"] = close.pct_change().fillna(0)
        features[f"{asset}__rsi"] = rsi / 100
        features[f"{asset}__macd_pct"] = macd / close
        features[f"{asset}__macd_signal_pct"] = signal / close
        features[f"{asset}__sma_gap"] = close / sma - 1

    result = pd.DataFrame(features, index=closes.index)
    if macro is not None:
        if macro.index.has_duplicates:
            raise ValueError("macro data must have unique dates")
        aligned_macro = macro.astype(float).sort_index().reindex(closes.index).ffill().shift(1)
        for name in aligned_macro.columns:
            values = aligned_macro[name]
            features_name = f"macro__{name}"
            result[features_name] = values.pct_change().replace([np.inf, -np.inf], np.nan)
            if result[features_name].isna().all():
                result[features_name] = 0.0
            else:
                result[features_name] = result[features_name].fillna(0)

    if sentiment is not None:
        if sentiment.index.has_duplicates:
            raise ValueError("sentiment must have unique dates")
        aligned_sentiment = sentiment.astype(float).sort_index().reindex(closes.index).ffill().shift(1)
        result["sentiment"] = aligned_sentiment.fillna(0).clip(-1, 1)

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("feature generation produced non-finite values")
    return result


def finbert_sentiment(
    articles: pd.DataFrame,
    date_column: str = "published_at",
    text_column: str = "text",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
) -> pd.Series:
    """Score news with FinBERT and return daily positive-minus-negative scores.

    Requires the ``sentiment`` extra. Dates are normalized to calendar days;
    ``build_features`` shifts the resulting series to prevent same-day leakage.
    """
    if date_column not in articles or text_column not in articles:
        raise ValueError(f"articles must contain {date_column!r} and {text_column!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    usable = articles[[date_column, text_column]].dropna().copy()
    if usable.empty:
        return pd.Series(dtype=float, name="sentiment")

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "FinBERT requires optional dependencies; install finance-project-rl[sentiment]"
        ) from exc

    scorer = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
        truncation=True,
        device=-1,
    )
    scores: list[float] = []
    texts = usable[text_column].astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batches = scorer(texts[start : start + batch_size], batch_size=batch_size)
        for labels in batches:
            by_label = {item["label"].lower(): float(item["score"]) for item in labels}
            scores.append(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))

    daily = pd.DataFrame(
        {
            "date": (
                pd.to_datetime(usable[date_column], errors="raise", utc=True)
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
                .dt.normalize()
                .to_numpy()
            ),
            "score": scores,
        }
    )
    return daily.groupby("date")["score"].mean().rename("sentiment")


def load_yahoo_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance."""
    if not symbols:
        raise ValueError("provide at least one ticker symbol")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Yahoo downloads require the optional 'data' extra") from exc

    raw = yf.download(
        tickers=list(symbols),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response does not contain close prices")
        closes = raw["Close"]
    elif len(symbols) == 1 and "Close" in raw.columns:
        closes = raw[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        raise ValueError("unexpected Yahoo Finance response format")
    closes = closes.reindex(columns=symbols)
    missing = closes.columns[closes.isna().all()].tolist()
    if missing:
        raise ValueError(f"Yahoo Finance returned no observations for: {', '.join(missing)}")
    closes = closes.dropna(how="all").ffill().dropna()
    if closes.empty:
        raise ValueError("no aligned close-price observations were returned")
    return closes
