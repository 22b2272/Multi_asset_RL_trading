"""Public data downloads for Indian-market research datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from finance_rl.features import load_yahoo_prices


def load_asset_universe(
    path: Path,
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Read a user-defined CSV universe of Yahoo tickers and MFAPI scheme codes."""
    universe = pd.read_csv(path, dtype=str).fillna("")
    required = {"asset_class", "ticker", "scheme_code"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"universe CSV is missing columns: {', '.join(sorted(missing))}")
    if universe.empty:
        raise ValueError("universe CSV must contain at least one asset")

    symbols: list[str] = []
    mutual_funds: dict[str, int] = {}
    classes: dict[str, str] = {}
    for row_number, row in universe.iterrows():
        asset_class = row["asset_class"].strip().lower()
        ticker = row["ticker"].strip()
        raw_scheme_code = row["scheme_code"].strip()
        label = row.get("label", "").strip()
        if asset_class not in {"stock", "etf", "gold_etf", "mutual_fund"}:
            raise ValueError(
                f"unsupported asset_class {asset_class!r} on universe CSV row {row_number + 2}; "
                "choose stock, etf, gold_etf, or mutual_fund"
            )
        if asset_class == "mutual_fund":
            if ticker or not raw_scheme_code:
                raise ValueError(
                    f"mutual_fund row {row_number + 2} requires scheme_code and blank ticker"
                )
            try:
                scheme_code = int(raw_scheme_code)
            except ValueError as exc:
                raise ValueError(
                    f"invalid MFAPI scheme_code on universe CSV row {row_number + 2}"
                ) from exc
            asset_label = label or str(scheme_code)
            if asset_label in mutual_funds:
                raise ValueError(f"duplicate mutual-fund label {asset_label!r}")
            mutual_funds[asset_label] = scheme_code
            classes[f"MF_{asset_label}"] = asset_class
        else:
            if not ticker or raw_scheme_code:
                raise ValueError(
                    f"{asset_class} row {row_number + 2} requires ticker and blank scheme_code"
                )
            if ticker in symbols:
                raise ValueError(f"duplicate Yahoo ticker {ticker!r}")
            symbols.append(ticker)
            classes[ticker] = asset_class
    if not symbols and not mutual_funds:
        raise ValueError("universe CSV must include at least one supported asset")
    return symbols, mutual_funds, classes


def fetch_mutual_fund_nav(
    scheme_code: int,
    start: str,
    end: str,
) -> tuple[str, pd.Series]:
    """Fetch historical mutual-fund NAV from MFAPI and filter the requested dates."""
    if scheme_code < 1:
        raise ValueError("scheme_code must be a positive integer")
    query = urlencode({"startDate": start, "endDate": end})
    payload = _get_json(f"https://api.mfapi.in/mf/{scheme_code}?{query}")
    if "data" not in payload or "meta" not in payload:
        raise ValueError(f"MFAPI returned an unexpected response for scheme {scheme_code}")
    values = pd.DataFrame(payload["data"])
    if values.empty:
        raise ValueError(f"MFAPI returned no NAV history for scheme {scheme_code}")
    dates = pd.to_datetime(values["date"], format="%d-%m-%Y", errors="raise")
    nav = pd.to_numeric(values["nav"], errors="raise")
    series = pd.Series(nav.to_numpy(), index=dates, name=f"MF_{scheme_code}").sort_index()
    series = series.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if series.empty:
        raise ValueError(f"MFAPI returned no NAV in {start} through {end} for scheme {scheme_code}")
    scheme_name = str(payload["meta"].get("scheme_name", f"MF {scheme_code}"))
    return scheme_name, series


def fetch_yahoo_news(symbols: Sequence[str], limit: int = 100) -> pd.DataFrame:
    """Fetch the latest Yahoo Finance RSS headlines; this is not a historical archive."""
    if not symbols or limit < 1:
        raise ValueError("symbols must be non-empty and limit must be positive")
    query = urlencode({"s": ",".join(symbols), "region": "IN", "lang": "en-IN"})
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?{query}"
    request = Request(url, headers={"User-Agent": "finance-project-rl/0.1"})
    with urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    articles = []
    for item in root.findall("./channel/item")[:limit]:
        title = item.findtext("title", default="")
        description = item.findtext("description", default="")
        articles.append(
            {
                "published_at": item.findtext("pubDate"),
                "title": title,
                "text": f"{title}. {description}".strip(". "),
                "link": item.findtext("link", default=""),
            }
        )
    return pd.DataFrame(articles, columns=["published_at", "title", "text", "link"])


def fetch_research_bundle(
    symbols: Sequence[str],
    mutual_funds: Mapping[str, int],
    macro_symbols: Sequence[str],
    start: str,
    end: str,
    output_dir: Path,
    include_news: bool = True,
    asset_classes: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Download and save asset prices/NAVs, macro market proxies, and latest headlines."""
    if not symbols and not mutual_funds:
        raise ValueError("provide at least one Yahoo Finance ticker or mutual-fund scheme")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_yahoo_symbols = list(dict.fromkeys([*symbols, *macro_symbols]))
    yahoo_prices = (
        load_yahoo_prices(all_yahoo_symbols, start, end)
        if all_yahoo_symbols
        else pd.DataFrame()
    )
    macro_columns = [symbol for symbol in macro_symbols if symbol in yahoo_prices]
    macro = yahoo_prices[macro_columns].rename(
        columns={symbol: symbol.replace("^", "").replace("=", "_") for symbol in macro_columns}
    )
    market = yahoo_prices[[symbol for symbol in symbols if symbol in yahoo_prices]].copy()
    fund_metadata: dict[str, str] = {}
    for label, scheme_code in mutual_funds.items():
        scheme_name, nav = fetch_mutual_fund_nav(scheme_code, start, end)
        fund_column = f"MF_{label}"
        market[fund_column] = nav.rename(fund_column)
        fund_metadata[fund_column] = scheme_name

    market = market.sort_index().ffill(limit=5).dropna(how="any")
    market.index.name = "date"
    macro = macro.sort_index().ffill().dropna(how="all")
    macro.index.name = "date"
    if market.empty:
        raise ValueError("no market data remained after aligning assets")

    outputs = {
        "prices": output_dir / "prices.csv",
        "macro": output_dir / "macro.csv",
        "metadata": output_dir / "metadata.json",
    }
    market.to_csv(outputs["prices"])
    macro.to_csv(outputs["macro"])
    outputs["metadata"].write_text(
        json.dumps(
            {
                "date_range": {"start": start, "end": end},
                "asset_sources": {
                    **{
                        symbol: {
                            "source": "Yahoo Finance adjusted close",
                            "asset_class": (asset_classes or {}).get(symbol, "stock"),
                        }
                        for symbol in symbols
                    },
                    **fund_metadata,
                    **{
                        name: {
                            "source": "MFAPI scheme history",
                            "asset_class": (asset_classes or {}).get(name, "mutual_fund"),
                        }
                        for name in fund_metadata
                    },
                    **{
                        f"macro__{column}": f"Yahoo Finance market proxy ({column})"
                        for column in macro.columns
                    },
                },
                "price_coverage": {
                    name: {
                        "start": str(market[name].first_valid_index().date()),
                        "end": str(market[name].last_valid_index().date()),
                        "observations": int(market[name].notna().sum()),
                    }
                    for name in market.columns
                },
                "mutual_fund_source": "MFAPI (scheme history data)",
                "news_source": "Yahoo Finance RSS (latest headlines only)",
                "news_is_historical": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if include_news:
        if symbols:
            outputs["news"] = output_dir / "news.csv"
            fetch_yahoo_news(symbols).to_csv(outputs["news"], index=False)
    return outputs


def _get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "finance-project-rl/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))
