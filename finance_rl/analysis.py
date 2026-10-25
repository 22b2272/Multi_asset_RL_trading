"""Descriptive portfolio analysis, comparison tables, and plot generation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from finance_rl.metrics import performance_metrics


def build_descriptive_statistics(
    prices: pd.DataFrame,
    asset_classes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize asset return, risk, and drawdown over the supplied price history."""
    if prices.empty or len(prices) < 2:
        raise ValueError("prices must contain at least two observations")
    if (prices <= 0).any().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("prices must contain finite, positive values")

    returns = prices.pct_change().dropna()
    rows = []
    for asset in prices.columns:
        asset_returns = returns[asset]
        wealth = (1 + asset_returns).cumprod()
        volatility = float(asset_returns.std(ddof=1) * np.sqrt(252))
        daily_std = float(asset_returns.std(ddof=1))
        rows.append(
            {
                "asset": asset,
                "asset_class": (asset_classes or {}).get(asset, "unspecified"),
                "start_price": float(prices[asset].iloc[0]),
                "end_price": float(prices[asset].iloc[-1]),
                "total_return": float(prices[asset].iloc[-1] / prices[asset].iloc[0] - 1),
                "annualized_return": float(
                    (prices[asset].iloc[-1] / prices[asset].iloc[0]) ** (252 / len(returns)) - 1
                ),
                "annualized_volatility": volatility,
                "sharpe_ratio_rf_0": (
                    float(asset_returns.mean() / daily_std * np.sqrt(252))
                    if daily_std > 0
                    else 0.0
                ),
                "sortino_ratio_rf_0": (
                    float(
                        asset_returns.mean()
                        / np.sqrt(np.mean(np.square(asset_returns[asset_returns < 0])))
                        * np.sqrt(252)
                    )
                    if (asset_returns < 0).any()
                    else 0.0
                ),
                "historical_var_95_daily": max(-float(asset_returns.quantile(0.05)), 0.0),
                "historical_cvar_95_daily": max(
                    -float(asset_returns[asset_returns <= asset_returns.quantile(0.05)].mean()),
                    0.0,
                ),
                "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
                "positive_day_fraction": float((asset_returns > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def generate_analysis_report(
    output_dir: Path,
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    agent_result: Mapping,
    baselines: Mapping[str, Mapping],
    asset_classes: Mapping[str, str] | None = None,
    training_returns: list[float] | None = None,
) -> dict[str, Path]:
    """Save financial tables, plots, and a Markdown report for a held-out run."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Analysis plots require matplotlib; install finance-project-rl[analysis]"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = build_descriptive_statistics(prices, asset_classes)
    returns = prices.pct_change().dropna()
    correlation = returns.corr()
    stats_path = output_dir / "asset_statistics.csv"
    correlation_path = output_dir / "asset_return_correlations.csv"
    stats.to_csv(stats_path)
    correlation.to_csv(correlation_path)

    agent_curve = np.asarray(agent_result["equity_curve"], dtype=float)
    if len(agent_curve) != len(dates):
        raise ValueError("dates and agent equity curve must have equal lengths")
    benchmark_curves = _baseline_curves(prices, baselines, float(agent_curve[0]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, agent_curve, label="RL agent", linewidth=2)
    for name, curve in benchmark_curves.items():
        ax.plot(dates, curve, label=name.replace("_", " ").title(), alpha=0.85)
    ax.set(title="Held-out portfolio value", ylabel="Portfolio value (INR)", xlabel="Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    equity_path = output_dir / "equity_curve.png"
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, curve in {"RL agent": agent_curve, **benchmark_curves}.items():
        drawdown = curve / np.maximum.accumulate(curve) - 1
        ax.plot(dates, drawdown * 100, label=name.replace("_", " ").title())
    ax.set(title="Held-out portfolio drawdown", ylabel="Drawdown (%)", xlabel="Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    drawdown_path = output_dir / "drawdown.png"
    fig.savefig(drawdown_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, max(5, len(stats) * 0.35)))
    colors = ["#2f855a" if value >= 0 else "#c53030" for value in stats["total_return"]]
    ax.barh(stats.index, stats["total_return"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Asset total return over analysis period", xlabel="Total return (%)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    asset_returns_path = output_dir / "asset_returns.png"
    fig.savefig(asset_returns_path, dpi=150)
    plt.close(fig)

    weights = pd.Series(agent_result["asset_weights"], dtype=float)
    weights.loc["Cash"] = float(agent_result["cash_weight"])
    weights = weights.sort_values()
    fig, ax = plt.subplots(figsize=(12, max(5, len(weights) * 0.35)))
    ax.barh(weights.index, weights.to_numpy() * 100, color="#3182ce")
    ax.set(title="Final RL portfolio allocation", xlabel="Portfolio weight (%)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    allocation_path = output_dir / "allocation.png"
    fig.savefig(allocation_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(correlation) * 0.7), max(6, len(correlation) * 0.6)))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation)), correlation.index)
    ax.set_title("Daily asset return correlation")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    correlation_plot_path = output_dir / "correlation_heatmap.png"
    fig.savefig(correlation_plot_path, dpi=150)
    plt.close(fig)

    daily_portfolio_returns = pd.Series(agent_curve, index=dates).pct_change()
    rolling_volatility = daily_portfolio_returns.rolling(30, min_periods=10).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, rolling_volatility * 100, color="#805ad5")
    ax.set(
        title="RL portfolio rolling 30-session annualized volatility",
        ylabel="Volatility (%)",
        xlabel="Date",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    rolling_risk_path = output_dir / "rolling_volatility.png"
    fig.savefig(rolling_risk_path, dpi=150)
    plt.close(fig)

    training_rewards_path = None
    if training_returns:
        fig, ax = plt.subplots(figsize=(9, 4))
        episodes = np.arange(1, len(training_returns) + 1)
        ax.plot(episodes, training_returns, marker="o", label="Episode reward")
        if len(training_returns) >= 3:
            rolling_window = min(5, len(training_returns))
            smoothed = pd.Series(training_returns).rolling(
                rolling_window, min_periods=1
            ).mean()
            ax.plot(episodes, smoothed, linewidth=2, label=f"{rolling_window}-episode mean")
            ax.legend()
        ax.set(title="Training reward by episode", ylabel="Total episode reward", xlabel="Episode")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        training_rewards_path = output_dir / "training_rewards.png"
        fig.savefig(training_rewards_path, dpi=150)
        plt.close(fig)

    markdown_path = output_dir / "financial_analysis.md"
    markdown_path.write_text(
        _report_markdown(
            dates,
            stats,
            correlation,
            agent_result,
            baselines,
            has_training_rewards=training_rewards_path is not None,
        ),
        encoding="utf-8",
    )
    outputs = {
        "asset_statistics": stats_path,
        "asset_return_correlations": correlation_path,
        "equity_curve": equity_path,
        "drawdown": drawdown_path,
        "asset_returns": asset_returns_path,
        "allocation": allocation_path,
        "correlation_heatmap": correlation_plot_path,
        "rolling_volatility": rolling_risk_path,
        "financial_analysis": markdown_path,
    }
    if training_rewards_path is not None:
        outputs["training_rewards"] = training_rewards_path
    return outputs


def _baseline_curves(
    prices: pd.DataFrame,
    baselines: Mapping[str, Mapping],
    starting_equity: float,
) -> dict[str, np.ndarray]:
    starting_prices = prices.iloc[0]
    initial_asset_value = starting_equity * (1 - 0.001) / len(prices.columns)
    growth = prices.div(starting_prices)
    equal_weight = initial_asset_value * growth.sum(axis=1).to_numpy()
    equal_weight = np.concatenate(([starting_equity], equal_weight[1:]))
    curves = {"equal_weight_buy_and_hold": equal_weight}
    if "cash" in baselines:
        curves["cash"] = np.full(len(prices), starting_equity)
    return curves


def _report_markdown(
    dates: pd.DatetimeIndex,
    stats: pd.DataFrame,
    correlation: pd.DataFrame,
    agent: Mapping,
    baselines: Mapping[str, Mapping],
    has_training_rewards: bool = False,
) -> str:
    benchmark_rows = {
        "RL agent": {
            "final_equity": agent["final_equity"],
            "total_return": agent["total_return"],
            "annualized_return": agent["annualized_return"],
            "annualized_volatility": agent["annualized_volatility"],
            "sharpe_ratio_rf_0": agent["sharpe_ratio_rf_0"],
            "sortino_ratio_rf_0": agent["sortino_ratio_rf_0"],
            "historical_var_95_daily": agent["historical_var_95_daily"],
            "historical_cvar_95_daily": agent["historical_cvar_95_daily"],
            "max_drawdown": agent["max_drawdown"],
        },
        **baselines,
    }
    benchmark_table = pd.DataFrame.from_dict(benchmark_rows, orient="index").to_markdown()
    top_returns = stats.sort_values("total_return", ascending=False).head(5)
    weakest_returns = stats.sort_values("total_return").head(5)
    strongest_correlations = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    ).stack()
    highest_pair = strongest_correlations.idxmax() if not strongest_correlations.empty else None
    highest_value = strongest_correlations.max() if highest_pair else None
    average_correlation = (
        f"{strongest_correlations.mean():.3f}"
        if not strongest_correlations.empty
        else "not available (fewer than two assets)"
    )
    highest_correlation = (
        f"{highest_pair[0]} / {highest_pair[1]} ({highest_value:.3f})"
        if highest_pair
        else "not available (fewer than two assets)"
    )

    return f"""# Financial Analysis Report

## Evaluation scope

- Period: {dates[0].date()} to {dates[-1].date()}
- Trading observations (including start): {len(dates)}
- Number of candidate assets: {len(stats)}
- Metrics are historical estimates, not forecasts or investment advice.
- Sharpe ratio assumes a zero risk-free rate; transaction costs for the RL
  agent are included in its equity curve.
- Daily 95% historical VaR/CVaR are loss magnitudes from the empirical lower
  tail; Sortino uses downside deviation, all with a zero risk-free assumption.

## Portfolio comparison

{benchmark_table}

## Asset statistics

Annualized return and volatility use 252 trading days. Sharpe uses a zero
risk-free rate. Maximum drawdown is computed from each asset's cumulative
adjusted price return.

{stats.to_markdown(floatfmt=".4f")}

## Diversification notes

- Average off-diagonal pairwise asset return correlation:
  {average_correlation}
- Highest pairwise correlation: {highest_correlation}
- Highest-return assets during this historical window:
  {", ".join(top_returns.index.tolist())}
- Lowest-return assets during this historical window:
  {", ".join(weakest_returns.index.tolist())}

## Plots

- ![Held-out equity curves](equity_curve.png)
- ![Drawdown](drawdown.png)
- ![Asset returns](asset_returns.png)
- ![Final allocation](allocation.png)
- ![Return correlations](correlation_heatmap.png)
- ![Rolling volatility](rolling_volatility.png)
{"- ![Training rewards](training_rewards.png)" if has_training_rewards else ""}
"""
