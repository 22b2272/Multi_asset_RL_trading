"""Portfolio performance metrics shared by agent and baseline evaluation."""

from __future__ import annotations

import numpy as np


def performance_metrics(equity_curve: list[float] | np.ndarray) -> dict[str, float]:
    """Calculate total/annualized return, volatility, Sharpe ratio, and drawdown."""
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.ndim != 1 or len(equity) < 2:
        raise ValueError("equity_curve must contain at least two observations")
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("equity_curve must contain finite, positive values")

    daily_returns = equity[1:] / equity[:-1] - 1
    total_return = float(equity[-1] / equity[0] - 1)
    annualized_return = float((equity[-1] / equity[0]) ** (252 / len(daily_returns)) - 1)
    volatility = (
        float(daily_returns.std(ddof=1) * np.sqrt(252))
        if len(daily_returns) > 1
        else 0.0
    )
    daily_volatility = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside_returns))))
        if len(downside_returns)
        else 0.0
    )
    return_quantile = float(np.quantile(daily_returns, 0.05))
    tail_returns = daily_returns[daily_returns <= return_quantile]
    sharpe = (
        float(daily_returns.mean() / daily_volatility * np.sqrt(252))
        if daily_volatility > 0
        else 0.0
    )
    sortino = (
        float(daily_returns.mean() / downside_deviation * np.sqrt(252))
        if downside_deviation > 0
        else 0.0
    )
    drawdowns = equity / np.maximum.accumulate(equity) - 1
    return {
        "final_equity": float(equity[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio_rf_0": sharpe,
        "sortino_ratio_rf_0": sortino,
        "historical_var_95_daily": max(-return_quantile, 0.0),
        "historical_cvar_95_daily": max(-float(tail_returns.mean()), 0.0),
        "max_drawdown": float(drawdowns.min()),
    }
