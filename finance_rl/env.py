"""Gymnasium environment for long-only, multi-asset portfolio trading."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class MultiAssetTradingEnv(gym.Env[np.ndarray, int]):
    """Trade a basket of assets using discrete hold/buy/sell actions.

    Action 0 holds. Actions 1..N buy a fraction of portfolio value in an asset;
    actions N+1..2N sell that fraction of the selected asset. Short selling is
    not supported. The observation is a lookback sequence with portfolio state
    appended to each row, suitable for recurrent Q-networks.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        lookback: int = 30,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        trade_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        if prices.empty or prices.columns.empty:
            raise ValueError("prices must contain at least one asset and one row")
        if not prices.index.equals(features.index):
            raise ValueError("prices and features must have identical date indexes")
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("prices must have unique dates and asset names")
        if not prices.columns.is_unique:
            raise ValueError("asset names must be unique")
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
            raise ValueError("prices must contain only finite, positive values")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("features must contain only finite values")
        if lookback < 1 or len(prices) <= lookback:
            raise ValueError("data must contain more rows than the positive lookback")
        if initial_cash <= 0 or not 0 <= transaction_cost < 1:
            raise ValueError("initial_cash must be positive and transaction_cost in [0, 1)")
        if not 0 < trade_fraction <= 1:
            raise ValueError("trade_fraction must be in (0, 1]")

        self.assets = list(prices.columns)
        self.prices = prices.to_numpy(dtype=np.float64)
        self.features = features.to_numpy(dtype=np.float32)
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.trade_fraction = float(trade_fraction)
        self.n_assets = len(self.assets)
        self.n_features = self.features.shape[1]
        self.action_space = spaces.Discrete(1 + 2 * self.n_assets)
        observation_size = self.n_features + self.n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, observation_size),
            dtype=np.float32,
        )
        self._shares = np.zeros(self.n_assets, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = self.lookback - 1
        self.cash = self.initial_cash
        self._shares.fill(0)
        self.equity = self.initial_cash
        self.total_costs = 0.0
        return self._observation(), self._info(0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.action_space.n - 1}]")
        if self.step_index >= len(self.prices) - 1:
            raise RuntimeError("step called after the episode ended; call reset()")

        current_prices = self.prices[self.step_index]
        previous_equity = self._mark_to_market(self.step_index)
        asset_values = self._shares * current_prices
        traded = 0.0
        if 1 <= action <= self.n_assets:
            asset = action - 1
            desired = min(
                previous_equity * self.trade_fraction,
                self.cash / (1 + self.transaction_cost),
            )
            self.cash -= desired * (1 + self.transaction_cost)
            self._shares[asset] += desired / current_prices[asset]
            traded = desired
        elif action > self.n_assets:
            asset = action - self.n_assets - 1
            desired = min(
                previous_equity * self.trade_fraction,
                asset_values[asset],
            )
            self._shares[asset] -= desired / current_prices[asset]
            self.cash += desired * (1 - self.transaction_cost)
            traded = desired

        cost = traded * self.transaction_cost
        self.total_costs += cost
        self.step_index += 1
        self.equity = self._mark_to_market(self.step_index)
        reward = float(np.log(max(self.equity, 1e-12) / max(previous_equity, 1e-12)))
        terminated = self.step_index >= len(self.prices) - 1
        return self._observation(), reward, terminated, False, self._info(cost)

    def _mark_to_market(self, index: int) -> float:
        return float(self.cash + (self._shares * self.prices[index]).sum())

    def _observation(self) -> np.ndarray:
        start = self.step_index - self.lookback + 1
        market = self.features[start : self.step_index + 1]
        equity = max(self._mark_to_market(self.step_index), 1e-12)
        asset_values = self._shares * self.prices[self.step_index]
        portfolio = np.concatenate(
            ([self.cash / equity], asset_values / equity)
        ).astype(np.float32)
        repeated_portfolio = np.broadcast_to(
            portfolio, (self.lookback, len(portfolio))
        )
        return np.concatenate((market, repeated_portfolio), axis=1).astype(np.float32)

    def _info(self, transaction_cost: float) -> dict:
        equity = max(self._mark_to_market(self.step_index), 1e-12)
        asset_values = self._shares * self.prices[self.step_index]
        return {
            "equity": equity,
            "cash": float(self.cash),
            "asset_values": asset_values.copy(),
            "portfolio_weights": (asset_values / equity).copy(),
            "total_costs": self.total_costs,
            "transaction_cost": transaction_cost,
            "step": self.step_index,
        }


class ContinuousPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Rebalance to continuous long-only asset and cash weights each step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        lookback: int = 30,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        super().__init__()
        if prices.empty or prices.columns.empty:
            raise ValueError("prices must contain at least one asset and one row")
        if not prices.index.equals(features.index):
            raise ValueError("prices and features must have identical date indexes")
        if prices.index.has_duplicates or prices.columns.has_duplicates:
            raise ValueError("prices must have unique dates and asset names")
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
            raise ValueError("prices must contain only finite, positive values")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("features must contain only finite values")
        if lookback < 1 or len(prices) <= lookback:
            raise ValueError("data must contain more rows than the positive lookback")
        if initial_cash <= 0 or not 0 <= transaction_cost < 1:
            raise ValueError("initial_cash must be positive and transaction_cost in [0, 1)")

        self.assets = list(prices.columns)
        self.prices = prices.to_numpy(dtype=np.float64)
        self.features = features.to_numpy(dtype=np.float32)
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.n_assets = len(self.assets)
        self.n_features = self.features.shape[1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, self.n_features + self.n_assets + 1),
            dtype=np.float32,
        )
        self._shares = np.zeros(self.n_assets, dtype=np.float64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = self.lookback - 1
        self.cash = self.initial_cash
        self._shares.fill(0)
        self.total_costs = 0.0
        return self._observation(), self._info(0.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        weights = np.asarray(action, dtype=np.float64)
        if weights.shape != self.action_space.shape or not np.isfinite(weights).all():
            raise ValueError(f"action must be a finite vector with shape {self.action_space.shape}")
        if (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("action weights must be non-negative with a positive sum")
        weights /= weights.sum()
        if self.step_index >= len(self.prices) - 1:
            raise RuntimeError("step called after the episode ended; call reset()")

        current_prices = self.prices[self.step_index]
        equity_before_trade = self._mark_to_market(self.step_index)
        current_values = self._shares * current_prices
        asset_weights = weights[:-1]
        net_equity = equity_before_trade
        transaction_cost = 0.0
        for _ in range(5):
            target_values = asset_weights * net_equity
            transaction_cost = float(
                np.abs(target_values - current_values).sum() * self.transaction_cost
            )
            net_equity = max(equity_before_trade - transaction_cost, 1e-12)

        target_values = asset_weights * net_equity
        self._shares = target_values / current_prices
        self.cash = max(net_equity - float(target_values.sum()), 0.0)
        self.total_costs += transaction_cost
        self.step_index += 1
        equity_after_move = self._mark_to_market(self.step_index)
        reward = float(
            np.log(max(equity_after_move, 1e-12) / max(equity_before_trade, 1e-12))
        )
        terminated = self.step_index >= len(self.prices) - 1
        return (
            self._observation(),
            reward,
            terminated,
            False,
            self._info(transaction_cost),
        )

    def _mark_to_market(self, index: int) -> float:
        return float(self.cash + (self._shares * self.prices[index]).sum())

    def _observation(self) -> np.ndarray:
        start = self.step_index - self.lookback + 1
        market = self.features[start : self.step_index + 1]
        equity = max(self._mark_to_market(self.step_index), 1e-12)
        values = self._shares * self.prices[self.step_index]
        weights = np.concatenate(([self.cash / equity], values / equity)).astype(np.float32)
        repeated_weights = np.broadcast_to(
            weights, (self.lookback, len(weights))
        )
        return np.concatenate((market, repeated_weights), axis=1).astype(np.float32)

    def _info(self, transaction_cost: float) -> dict:
        equity = max(self._mark_to_market(self.step_index), 1e-12)
        values = self._shares * self.prices[self.step_index]
        return {
            "equity": equity,
            "cash": float(self.cash),
            "asset_values": values.copy(),
            "portfolio_weights": (values / equity).copy(),
            "cash_weight": float(self.cash / equity),
            "total_costs": self.total_costs,
            "transaction_cost": transaction_cost,
            "step": self.step_index,
        }
