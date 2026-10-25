"""Command-line interface for data preparation and RL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance_rl.env import ContinuousPortfolioEnv, MultiAssetTradingEnv
from finance_rl.features import build_features, finbert_sentiment, load_yahoo_prices
from finance_rl.metrics import performance_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an RL agent on Indian-market assets.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default="2014-10-04")
    parser.add_argument("--end", default="2024-10-04")
    parser.add_argument("--prices-csv", type=Path)
    parser.add_argument("--macro-csv", type=Path)
    parser.add_argument("--news-csv", type=Path)
    parser.add_argument("--agent", choices=["dqn", "ddqn", "ppo"], default="ppo")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser


def _baselines(prices: pd.DataFrame, lookback: int, initial_cash: float = 100_000.0) -> dict:
    selected_prices = prices.iloc[lookback - 1 :]
    first_prices = selected_prices.iloc[0]
    invested_values = (
        selected_prices.div(first_prices)
        * (initial_cash * (1 - 0.001) / len(prices.columns))
    )
    equal_weight_curve = np.concatenate(
        ([initial_cash], invested_values.iloc[1:].sum(axis=1).to_numpy())
    )
    cash_curve = np.full(len(equal_weight_curve), initial_cash)
    return {
        "cash": performance_metrics(cash_curve),
        "equal_weight_buy_and_hold": performance_metrics(equal_weight_curve),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.prices_csv:
        price_data = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    else:
        if not args.symbols:
            raise ValueError("provide --prices-csv or select assets with --symbols")
        price_data = load_yahoo_prices(args.symbols, args.start, args.end)
    price_data = price_data.sort_index()
    macro = (
        pd.read_csv(args.macro_csv, index_col=0, parse_dates=True)
        if args.macro_csv
        else None
    )
    sentiment = None
    if args.news_csv:
        sentiment = finbert_sentiment(pd.read_csv(args.news_csv))
        if sentiment.index.intersection(price_data.index.normalize()).empty:
            raise ValueError(
                "news dates do not overlap the price history; current RSS headlines "
                "cannot be used as historical training sentiment"
            )
    features = build_features(price_data, macro=macro, sentiment=sentiment)
    split_index = int(len(price_data) * args.train_fraction)
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be strictly between 0 and 1")
    if split_index <= args.lookback or len(price_data) - split_index < 2:
        raise ValueError("price data is too short for the requested lookback and train fraction")
    train_prices = price_data.iloc[:split_index]
    train_features = features.iloc[:split_index]
    test_start = split_index - args.lookback
    test_prices = price_data.iloc[test_start:]
    test_features = features.iloc[test_start:]
    environment = ContinuousPortfolioEnv if args.agent == "ppo" else MultiAssetTradingEnv
    env = environment(train_prices, train_features, lookback=args.lookback)
    test_env = environment(test_prices, test_features, lookback=args.lookback)

    from finance_rl.agents import TrainingConfig, evaluate_agent, train_dqn, train_ppo

    config = TrainingConfig(episodes=args.episodes, seed=args.seed)
    if args.agent == "ppo":
        model, returns = train_ppo(env, config)
    else:
        model, returns = train_dqn(env, config, algorithm=args.agent)
    evaluation = evaluate_agent(test_env, model, args.agent)
    benchmarks = _baselines(test_prices, args.lookback)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"{args.agent}_model.pt"
    import torch

    torch.save(model.state_dict(), model_path)
    pd.Series(returns, name="episode_return").to_csv(
        args.output / f"{args.agent}_returns.csv", index_label="episode"
    )
    (args.output / f"{args.agent}_evaluation.json").write_text(
        json.dumps({"agent": evaluation, "baselines": benchmarks}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model to {model_path}")
    print(f"Final training episode reward: {returns[-1]:.6f}")
    print(f"Held-out final equity: {evaluation['final_equity']:.2f}")
    print(f"Equal-weight buy-and-hold equity: {benchmarks['equal_weight_buy_and_hold']['final_equity']:.2f}")


if __name__ == "__main__":
    main()
