import numpy as np
import pandas as pd
import pytest

from finance_rl.data_sources import load_asset_universe
from finance_rl.env import ContinuousPortfolioEnv, MultiAssetTradingEnv
from finance_rl.features import build_features
from finance_rl.metrics import performance_metrics
from finance_rl.analysis import build_descriptive_statistics, generate_analysis_report


@pytest.fixture
def market_data():
    dates = pd.date_range("2024-01-01", periods=45, freq="B")
    prices = pd.DataFrame(
        {
            "RELIANCE.NS": np.linspace(100, 120, len(dates)),
            "TCS.NS": np.linspace(200, 180, len(dates)),
        },
        index=dates,
    )
    macro = pd.DataFrame({"india_vix": np.linspace(15, 18, len(dates))}, index=dates)
    sentiment = pd.Series(np.linspace(-0.5, 0.5, len(dates)), index=dates)
    return prices, macro, sentiment


def test_feature_builder_aligns_macro_and_lags_sentiment(market_data):
    prices, macro, sentiment = market_data
    features = build_features(prices, macro, sentiment)
    assert len(features) == len(prices)
    assert "RELIANCE.NS__rsi" in features
    assert "TCS.NS__macd_pct" in features
    assert features["sentiment"].iloc[0] == 0
    assert features["sentiment"].iloc[1] == pytest.approx(sentiment.iloc[0])
    assert np.isfinite(features.to_numpy()).all()


def test_environment_observation_actions_and_portfolio_info(market_data):
    prices, macro, sentiment = market_data
    features = build_features(prices, macro, sentiment)
    env = MultiAssetTradingEnv(prices, features, lookback=5, transaction_cost=0)
    observation, info = env.reset(seed=1)
    assert observation.shape == (5, features.shape[1] + 3)
    assert env.action_space.n == 5
    assert info["equity"] == pytest.approx(100_000)

    observation, reward, terminated, truncated, info = env.step(1)
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["portfolio_weights"][0] > 0
    assert info["equity"] > 100_000


def test_environment_rejects_misaligned_data(market_data):
    prices, _, _ = market_data
    with pytest.raises(ValueError, match="identical date indexes"):
        MultiAssetTradingEnv(prices, build_features(prices.iloc[1:]), lookback=5)


def test_environment_charges_trade_costs_and_finishes_episode(market_data):
    prices, _, _ = market_data
    features = build_features(prices)
    env = MultiAssetTradingEnv(prices, features, lookback=5)
    env.reset(seed=4)

    _, _, terminated, _, info = env.step(1)
    assert not terminated
    assert info["transaction_cost"] == pytest.approx(10)
    assert info["total_costs"] == pytest.approx(10)
    assert info["cash"] >= 0

    while not terminated:
        _, _, terminated, _, _ = env.step(0)
    with pytest.raises(RuntimeError, match="after the episode ended"):
        env.step(0)


def test_continuous_environment_rebalances_to_asset_and_cash_weights(market_data):
    prices, macro, sentiment = market_data
    features = build_features(prices, macro, sentiment)
    env = ContinuousPortfolioEnv(prices, features, lookback=5, transaction_cost=0)
    observation, _ = env.reset(seed=3)
    assert env.action_space.shape == (len(prices.columns) + 1,)
    assert observation.shape == env.observation_space.shape

    allocation = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    _, reward, terminated, _, info = env.step(allocation)
    assert not terminated
    assert np.isfinite(reward)
    expected_equity = 100_000 * (
        0.4 * prices.iloc[5, 0] / prices.iloc[4, 0]
        + 0.3 * prices.iloc[5, 1] / prices.iloc[4, 1]
        + 0.3
    )
    assert info["cash"] == pytest.approx(30_000, abs=1e-3)
    assert info["equity"] == pytest.approx(expected_equity, rel=1e-6)


def test_continuous_environment_rejects_invalid_allocations(market_data):
    prices, _, _ = market_data
    env = ContinuousPortfolioEnv(prices, build_features(prices), lookback=5)
    env.reset(seed=3)
    with pytest.raises(ValueError, match="positive sum"):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_dqn_and_ppo_can_train_and_evaluate(market_data):
    pytest.importorskip("torch")
    from finance_rl.agents import TrainingConfig, evaluate_agent, train_dqn, train_ppo

    prices, _, _ = market_data
    features = build_features(prices)
    config = TrainingConfig(
        episodes=1,
        batch_size=4,
        replay_capacity=10,
        target_update=2,
        hidden_size=8,
    )

    discrete_env = MultiAssetTradingEnv(prices.iloc[:12], features.iloc[:12], lookback=3)
    dqn, dqn_returns = train_dqn(discrete_env, config, algorithm="ddqn")
    assert len(dqn_returns) == 1
    assert evaluate_agent(discrete_env, dqn, "ddqn")["final_equity"] > 0

    continuous_env = ContinuousPortfolioEnv(prices.iloc[:12], features.iloc[:12], lookback=3)
    ppo, ppo_returns = train_ppo(continuous_env, config, epochs=1)
    assert len(ppo_returns) == 1
    assert evaluate_agent(continuous_env, ppo, "ppo")["final_equity"] > 0


def test_user_defined_universe_supports_multiple_asset_classes(tmp_path):
    universe_file = tmp_path / "universe.csv"
    universe_file.write_text(
        "asset_class,ticker,scheme_code,label\n"
        "stock,RELIANCE.NS,,\n"
        "gold_etf,GOLDBEES.NS,,\n"
        "mutual_fund,,118347,Index_Fund\n",
        encoding="utf-8",
    )

    symbols, funds, asset_classes = load_asset_universe(universe_file)

    assert symbols == ["RELIANCE.NS", "GOLDBEES.NS"]
    assert funds == {"Index_Fund": 118347}
    assert asset_classes["RELIANCE.NS"] == "stock"
    assert asset_classes["GOLDBEES.NS"] == "gold_etf"
    assert asset_classes["MF_Index_Fund"] == "mutual_fund"


def test_user_defined_universe_rejects_derivatives_until_supported(tmp_path):
    universe_file = tmp_path / "universe.csv"
    universe_file.write_text(
        "asset_class,ticker,scheme_code\nfuture,NIFTY-FUT,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported asset_class"):
        load_asset_universe(universe_file)


def test_performance_metrics_calculates_return_and_drawdown():
    metrics = performance_metrics([100, 120, 90, 110])
    assert metrics["final_equity"] == pytest.approx(110)
    assert metrics["total_return"] == pytest.approx(0.1)
    assert metrics["max_drawdown"] == pytest.approx(-0.25)


def test_descriptive_statistics_reports_risk_and_asset_classes(market_data):
    prices, _, _ = market_data
    statistics = build_descriptive_statistics(
        prices,
        {"RELIANCE.NS": "stock", "TCS.NS": "mutual_fund"},
    )
    assert statistics.loc["RELIANCE.NS", "asset_class"] == "stock"
    assert statistics.loc["TCS.NS", "asset_class"] == "mutual_fund"
    assert statistics.loc["RELIANCE.NS", "annualized_volatility"] >= 0
    assert statistics.loc["RELIANCE.NS", "max_drawdown"] <= 0


def test_analysis_report_writes_tables_markdown_and_plots(tmp_path, market_data):
    pytest.importorskip("matplotlib")
    prices, _, _ = market_data
    dates = prices.index[4:]
    agent = {
        "equity_curve": np.linspace(100_000, 110_000, len(dates)).tolist(),
        "final_equity": 110_000.0,
        "total_return": 0.1,
        "annualized_return": 0.1,
        "annualized_volatility": 0.1,
        "sharpe_ratio_rf_0": 1.0,
        "sortino_ratio_rf_0": 1.0,
        "historical_var_95_daily": 0.02,
        "historical_cvar_95_daily": 0.03,
        "max_drawdown": -0.02,
        "asset_weights": {"RELIANCE.NS": 0.45, "TCS.NS": 0.45},
        "cash_weight": 0.1,
    }
    baselines = {
        "cash": performance_metrics(np.full(len(dates), 100_000.0)),
        "equal_weight_buy_and_hold": performance_metrics(np.linspace(100_000, 108_000, len(dates))),
    }
    outputs = generate_analysis_report(
        tmp_path,
        dates,
        prices.iloc[4:],
        agent,
        baselines,
        training_returns=[0.1, 0.2, 0.3],
    )
    assert len(outputs) == 10
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
