"""Optional PyTorch DQN/DDQN-LSTM and PPO training implementations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Literal

import numpy as np

from finance_rl.metrics import performance_metrics

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "RL training requires PyTorch; install finance-project-rl[deep-learning]"
    ) from exc


class QLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_count: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(observations)
        return self.head(sequence[:, -1])


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.shared(observations)
        concentrations = nn.functional.softplus(self.actor(latent)) + 1.0
        return concentrations, self.critic(latent).squeeze(-1)


@dataclass
class TrainingConfig:
    episodes: int = 20
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 20_000
    target_update: int = 250
    hidden_size: int = 64
    seed: int = 7

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("episodes and batch_size must be positive, and replay_capacity >= batch_size")
        if self.target_update < 1 or self.hidden_size < 1 or self.learning_rate <= 0:
            raise ValueError("target_update, hidden_size, and learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")


def train_dqn(
    env,
    config: TrainingConfig | None = None,
    algorithm: Literal["dqn", "ddqn"] = "ddqn",
) -> tuple[QLSTM, list[float]]:
    """Train a recurrent DQN or Double-DQN agent on one environment."""
    if algorithm not in {"dqn", "ddqn"}:
        raise ValueError("algorithm must be 'dqn' or 'ddqn'")
    config = config or TrainingConfig()
    _seed(config.seed)
    env.action_space.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.observation_space.shape[-1]
    action_count = env.action_space.n
    online = QLSTM(input_size, config.hidden_size, action_count).to(device)
    target = QLSTM(input_size, config.hidden_size, action_count).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    memory: deque = deque(maxlen=config.replay_capacity)
    episode_returns: list[float] = []
    transitions = 0

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        done = False
        total_reward = 0.0
        while not done:
            epsilon = max(0.02, 1.0 - transitions / max(1, config.episodes * 500))
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online(state).argmax(dim=1).item())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((observation, action, reward, next_observation, done))
            observation = next_observation
            total_reward += reward
            transitions += 1

            if len(memory) >= config.batch_size:
                batch = random.sample(memory, config.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
                next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)
                predicted = online(states_t).gather(1, actions_t[:, None]).squeeze(1)
                with torch.no_grad():
                    if algorithm == "ddqn":
                        next_actions = online(next_states_t).argmax(dim=1, keepdim=True)
                        next_values = target(next_states_t).gather(1, next_actions).squeeze(1)
                    else:
                        next_values = target(next_states_t).max(dim=1).values
                    expected = rewards_t + config.gamma * next_values * (1 - dones_t)
                loss = nn.functional.smooth_l1_loss(predicted, expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=1.0)
                optimizer.step()

            if transitions % config.target_update == 0:
                target.load_state_dict(online.state_dict())
        episode_returns.append(total_reward)
    return online, episode_returns


def train_ppo(
    env,
    config: TrainingConfig | None = None,
    epochs: int = 4,
    clip_ratio: float = 0.2,
) -> tuple[ActorCritic, list[float]]:
    """Train a clipped PPO actor-critic using complete episode rollouts."""
    config = config or TrainingConfig()
    if epochs < 1 or not 0 < clip_ratio < 1:
        raise ValueError("epochs must be positive and clip_ratio in (0, 1)")
    _seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flattened_size = int(np.prod(env.observation_space.shape))
    action_count = int(env.action_space.shape[0])
    model = ActorCritic(flattened_size, action_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    episode_returns: list[float] = []

    for episode in range(config.episodes):
        observations, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        observation, _ = env.reset(seed=config.seed + episode)
        done = False
        total_reward = 0.0
        while not done:
            state = torch.as_tensor(
                observation.reshape(-1), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                concentrations, value = model(state)
                distribution = torch.distributions.Dirichlet(concentrations)
                action = distribution.sample()
            action_value = action.squeeze(0).cpu().numpy()
            next_observation, reward, terminated, truncated, _ = env.step(action_value)
            done = terminated or truncated
            observations.append(observation.reshape(-1))
            actions.append(action_value)
            log_probs.append(float(distribution.log_prob(action).item()))
            rewards.append(float(reward))
            values.append(float(value.item()))
            dones.append(done)
            observation = next_observation
            total_reward += reward

        with torch.no_grad():
            next_state = torch.as_tensor(
                observation.reshape(-1), dtype=torch.float32, device=device
            ).unsqueeze(0)
            _, next_value_tensor = model(next_state)
            next_value = float(next_value_tensor.item())
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            continuation = 0.0 if dones[index] else 1.0
            following_value = next_value if index == len(rewards) - 1 else values[index + 1]
            delta = rewards[index] + config.gamma * following_value * continuation - values[index]
            gae = delta + config.gamma * 0.95 * continuation * gae
            advantages[index] = gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        for _ in range(epochs):
            indices = torch.randperm(len(actions_t), device=device)
            for start in range(0, len(actions_t), config.batch_size):
                batch_indices = indices[start : start + config.batch_size]
                concentrations, predicted_values = model(states_t[batch_indices])
                distribution = torch.distributions.Dirichlet(concentrations)
                new_log_probs = distribution.log_prob(actions_t[batch_indices])
                ratio = (new_log_probs - old_log_probs[batch_indices]).exp()
                unclipped = ratio * advantages_t[batch_indices]
                clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages_t[batch_indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(predicted_values, returns_t[batch_indices])
                entropy_loss = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
        episode_returns.append(total_reward)
    return model, episode_returns


def evaluate_agent(env, model: nn.Module, algorithm: Literal["dqn", "ddqn", "ppo"]) -> dict:
    """Run one deterministic episode and summarize held-out performance."""
    if algorithm not in {"dqn", "ddqn", "ppo"}:
        raise ValueError("algorithm must be 'dqn', 'ddqn', or 'ppo'")
    device = next(model.parameters()).device
    observation, initial_info = env.reset()
    equity_curve = [float(initial_info["equity"])]
    total_reward = 0.0
    done = False
    while not done:
        state_array = observation if algorithm != "ppo" else observation.reshape(-1)
        state = torch.as_tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if algorithm == "ppo":
                concentrations, _ = model(state)
                action = torch.distributions.Dirichlet(concentrations).mean.squeeze(0).cpu().numpy()
            else:
                action = int(model(state).argmax(dim=1).item())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        equity_curve.append(float(info["equity"]))
        done = terminated or truncated
    return {
        **performance_metrics(equity_curve),
        "total_log_return": total_reward,
        "total_costs": float(info["total_costs"]),
        "cash_weight": float(info.get("cash_weight", info["cash"] / info["equity"])),
        "equity_curve": equity_curve,
        "asset_weights": {
            asset: float(weight)
            for asset, weight in zip(env.assets, info["portfolio_weights"])
        },
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
