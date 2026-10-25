"""Download sample public Indian-market data into local CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from finance_rl.data_sources import fetch_research_bundle, load_asset_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Indian-market research data.")
    parser.add_argument("--start", default="2014-10-04")
    parser.add_argument("--end", default="2024-10-04")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Yahoo Finance tickers, such as any supported Indian stock or ETF",
    )
    parser.add_argument(
        "--funds",
        nargs="+",
        type=int,
        help="MFAPI mutual-fund scheme codes",
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        help="CSV with asset_class,ticker,scheme_code and optional label columns",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="skip Yahoo Finance RSS (latest headlines only; not historical training data)",
    )
    args = parser.parse_args()
    if args.universe_csv and (args.symbols or args.funds):
        parser.error("--universe-csv cannot be combined with --symbols or --funds")
    asset_classes = None
    if args.universe_csv:
        symbols, mutual_funds, asset_classes = load_asset_universe(args.universe_csv)
    else:
        symbols = args.symbols or []
        mutual_funds = {str(code): code for code in (args.funds or [])}
        if not symbols and not mutual_funds:
            parser.error("provide --universe-csv, --symbols, or --funds")
    outputs = fetch_research_bundle(
        symbols=symbols,
        mutual_funds=mutual_funds,
        macro_symbols=["^INDIAVIX", "USDINR=X", "CL=F"],
        start=args.start,
        end=args.end,
        output_dir=args.output,
        include_news=not args.no_news,
        asset_classes=asset_classes,
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
