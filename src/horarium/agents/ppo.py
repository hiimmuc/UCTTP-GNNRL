"""Single-file PPO with action masking, adapted from CleanRL's `ppo.py`.

Stable-Baselines3 is not usable here: the action space is (course x period) and its legal
subset changes every step and every instance, which SB3's fixed-space assumption rules out.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from horarium.agents._obs import edge_tensors, encoder_input, period_batch
from horarium.envs.base import EpisodeStats, TimetablingEnv
from horarium.models.encoders import Encoder, EncoderInput
from horarium.models.heads import ActionHead, ValueHead
from horarium.models.masking import masked_categorical

__all__ = [
    "RECENT_EPISODES",
    "ActorCritic",
    "PPOConfig",
    "TrainingReport",
    "resolve_device",
    "train",
]

#: Window the reported feasibility rate looks back over.
RECENT_EPISODES = 50


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters. Defaults are CleanRL's, with a rollout short enough for one instance."""

    total_steps: int = 50_000
    rollout_steps: int = 512
    update_epochs: int = 4
    minibatches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cpu"
    max_minibatch_edges: int = 600_000
    """Cap on `chunk x conflict edges`, which is what the edge-conditioned encoders allocate.

    An MPNN layer materialises one `(chunk, n_edges, dim)` tensor per MLP stage and keeps them
    all for the backward pass. On the Erlangen instances (24k edges) a slice of 256 asks for
    several GB and the run dies of CUDA OOM.

    This bounds only how much is put through the network at once. The minibatch stays
    `rollout_steps // minibatches` and gradients are accumulated across chunks before the
    optimiser steps, so effective batch size -- and with it gradient noise and the effective
    learning rate -- is identical whether or not the cap binds. It used to shrink the minibatch
    itself, which made the optimiser quietly differ by instance and by encoder on exactly the
    six Erlangen instances, confounding the arm it constrained most.
    """


@dataclass(slots=True)
class TrainingReport:
    """What a training run produced, enough to judge it without reading the logs."""

    updates: int
    steps: int
    episodes: int
    feasible_episodes: int
    best_cost: float
    best_stats: EpisodeStats | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    agent: ActorCritic | None = None
    """The trained network, so the caller can checkpoint or immediately run inference with it."""
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=RECENT_EPISODES))

    @property
    def feasibility_rate(self) -> float:
        """Fraction of all finished episodes that were feasible. Lags badly while learning."""
        return self.feasible_episodes / self.episodes if self.episodes else 0.0

    @property
    def recent_feasibility_rate(self) -> float:
        """Feasibility over the last RECENT_EPISODES episodes: what the agent does *now*."""
        return sum(self.recent) / len(self.recent) if self.recent else 0.0


class ActorCritic(nn.Module):
    """An encoder plus the two heads. Swapping the encoder changes nothing else here."""

    def __init__(self, encoder: Encoder, period_features: int) -> None:
        """Wire an encoder to a fresh action head and value head.

        Args:
            encoder: The state encoder; must also be an `nn.Module`.
            period_features: Width of the per-period feature vector.

        Raises:
            TypeError: `encoder` is not an `nn.Module`, so it has no parameters to optimise.
        """
        super().__init__()
        if not isinstance(encoder, nn.Module):  # pragma: no cover - guards a wiring mistake
            msg = "encoder must be an nn.Module so its parameters are optimised"
            raise TypeError(msg)
        self.encoder = encoder
        self.action_head = ActionHead(encoder.embedding_dim, period_features)
        self.value_head = ValueHead(encoder.embedding_dim)

    def forward(
        self, batch: EncoderInput, periods: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of observations.

        Args:
            batch: The node, global and graph-structure features to encode.
            periods: Per-period features, shape `(batch, n_periods, period_features)`.

        Returns:
            Flat `(batch, n_courses * n_periods)` action logits and `(batch,)` state values.
        """
        node_embeddings, graph_embedding = self.encoder(batch)
        return (
            self.action_head(node_embeddings, graph_embedding, periods),
            self.value_head(graph_embedding),
        )


def resolve_device(name: str) -> torch.device:
    """Turn a device name into a device, refusing to fall back to the CPU behind the caller's back.

    A torch wheel built for a newer CUDA than the driver installs happily and then reports
    `cuda.is_available() == False` with only a warning, so asking for the GPU and silently
    getting the CPU is a real failure mode here.

    Args:
        name: A torch device name, e.g. `"cpu"` or `"cuda"`.

    Returns:
        The resolved device.

    Raises:
        RuntimeError: CUDA was asked for and torch cannot use it.
    """
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = (
            f"device {name!r} requested but torch cannot use CUDA. This torch is built for CUDA "
            f"{torch.version.cuda}; check it matches the driver (`nvidia-smi`), and see the "
            f"cu128 index pinned in pyproject.toml."
        )
        raise RuntimeError(msg)
    return device


class _Rollout:
    """Fixed-size storage for one rollout of a single environment."""

    def __init__(self, steps: int, env: TimetablingEnv, device: torch.device) -> None:
        """Allocate every buffer up front so the update never reallocates.

        Args:
            steps: Rollout length; the size of every buffer's leading dimension.
            env: The environment being rolled out, used only for its shapes.
            device: Device the buffers are allocated on.
        """
        self.nodes = torch.zeros(
            (steps, env.instance.n_courses, env.node_feature_dim), device=device
        )
        self.periods = torch.zeros(
            (steps, env.instance.n_periods, env.period_feature_dim), device=device
        )
        self.globals = torch.zeros((steps, 1, env.global_feature_dim), device=device)
        self.masks = torch.zeros((steps, env.n_actions), dtype=torch.bool, device=device)
        self.actions = torch.zeros(steps, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(steps, device=device)
        self.rewards = torch.zeros(steps, device=device)
        self.dones = torch.zeros(steps, device=device)
        self.values = torch.zeros(steps, device=device)


def train(
    env: TimetablingEnv,
    build_encoder: Callable[[], Encoder],
    config: PPOConfig,
    *,
    on_update: Callable[[TrainingReport], None] | None = None,
) -> TrainingReport:
    """Run PPO on one environment and return what it achieved.

    Actions are sampled from the environment's mask, so every rollout state -- and therefore
    every solution the agent produces -- satisfies the hard constraints by construction.

    The encoder arrives as a factory, not an instance, so its weights are drawn after the seed
    is set. Accepting a built encoder made the run depend on whatever torch's global RNG state
    happened to be, which silently broke reproducibility.

    Args:
        env: The environment to train on.
        build_encoder: Factory returning a fresh, unseeded-until-now encoder.
        config: Hyperparameters for the run.
        on_update: Called with the report after every PPO update, for progress reporting. It
            must not mutate the report; nothing in the training loop reads its result.

    Returns:
        What the run produced, including the trained agent.

    Raises:
        TypeError: `build_encoder` is an encoder rather than a factory that returns one.
    """
    if isinstance(build_encoder, nn.Module):
        msg = "pass an encoder factory, not an encoder: weights must be drawn after seeding"
        raise TypeError(msg)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    agent = ActorCritic(build_encoder(), env.period_feature_dim).to(device)
    optimiser = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    edge_index, edge_attr = edge_tensors(env, device)

    storage = _Rollout(config.rollout_steps, env, device)
    report = TrainingReport(
        updates=0, steps=0, episodes=0, feasible_episodes=0, best_cost=float("inf")
    )

    observation, _ = env.reset(seed=config.seed)
    updates = max(1, config.total_steps // config.rollout_steps)

    for update in range(updates):
        returns_seen: list[float] = []
        for step in range(config.rollout_steps):
            batch = encoder_input(observation, edge_index, edge_attr, device)
            periods = period_batch(observation, device)
            mask = torch.as_tensor(observation["action_mask"], device=device).bool()
            with torch.no_grad():
                logits, value = agent(batch, periods)
                distribution = masked_categorical(logits, mask.unsqueeze(0))
                action = distribution.sample()  # type: ignore[no-untyped-call]

            storage.nodes[step] = batch.nodes[0]
            storage.periods[step] = periods[0]
            storage.globals[step] = batch.globals[0]
            storage.masks[step] = mask
            storage.actions[step] = action[0]
            storage.log_probs[step] = distribution.log_prob(action)[0]  # type: ignore[no-untyped-call]
            storage.values[step] = value[0]

            observation, reward, terminated, truncated, info = env.step(np.int64(action.item()))
            storage.rewards[step] = float(reward)
            storage.dones[step] = float(terminated or truncated)
            report.steps += 1

            if terminated or truncated:
                _record_episode(report, info["episode_stats"], returns_seen)
                observation, _ = env.reset()

        with torch.no_grad():
            batch = encoder_input(observation, edge_index, edge_attr, device)
            _, bootstrap = agent(batch, period_batch(observation, device))
        advantages, targets = _gae(storage, bootstrap[0], config)
        _optimise(
            agent, optimiser, storage, advantages, targets, config, edge_index, edge_attr, rng
        )

        report.updates = update + 1
        report.history.append(
            {
                "update": float(update + 1),
                "steps": float(report.steps),
                "episodes": float(report.episodes),
                "feasibility_rate": report.feasibility_rate,
                "recent_feasibility_rate": report.recent_feasibility_rate,
                "mean_episode_cost": (
                    float(np.mean(returns_seen)) if returns_seen else float("nan")
                ),
                "best_cost": report.best_cost,
            }
        )
        if on_update is not None:
            on_update(report)
    report.agent = agent
    return report


def _record_episode(report: TrainingReport, stats: EpisodeStats, costs: list[float]) -> None:
    """Fold one finished episode into the running report.

    Args:
        report: The report to update in place.
        stats: What the finished episode achieved.
        costs: This update's running list of episode costs, appended to in place.
    """
    report.episodes += 1
    report.feasible_episodes += int(stats.feasible)
    report.recent.append(stats.feasible)
    costs.append(stats.total_cost)
    if stats.feasible and stats.total_cost < report.best_cost:
        report.best_cost = stats.total_cost
        report.best_stats = stats


def _gae(
    storage: _Rollout, bootstrap: torch.Tensor, config: PPOConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over the stored rollout.

    `dones[t]` records that step `t` ended its episode, so the bootstrap after step `t` is
    masked by `1 - dones[t]`. Carrying the previous iteration's flag instead -- the shape this
    loop had first -- leaks value across episode boundaries and smears the terminal dead-end
    penalty into the next episode, which is exactly the signal the agent has to learn from.

    Args:
        storage: The completed rollout.
        bootstrap: State value of the step just past the end of the rollout.
        config: Hyperparameters, for `gamma` and `gae_lambda`.

    Returns:
        Advantages and value targets, one per rollout step.
    """
    advantages = torch.zeros_like(storage.rewards)
    running = torch.zeros((), device=storage.rewards.device)
    next_value = bootstrap
    for step in reversed(range(storage.rewards.shape[0])):
        non_terminal = 1.0 - storage.dones[step]
        residual = (
            storage.rewards[step] + config.gamma * next_value * non_terminal - storage.values[step]
        )
        running = residual + config.gamma * config.gae_lambda * non_terminal * running
        advantages[step] = running
        next_value = storage.values[step]
    return advantages, advantages + storage.values


def _optimise(  # noqa: PLR0913, PLR0917 - PPO's update genuinely needs all of these
    agent: ActorCritic,
    optimiser: torch.optim.Optimizer,
    storage: _Rollout,
    advantages: torch.Tensor,
    targets: torch.Tensor,
    config: PPOConfig,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    rng: np.random.Generator,
) -> None:
    """The clipped-surrogate update, over shuffled minibatches for a few epochs.

    Args:
        agent: The network being trained, updated in place.
        optimiser: Optimiser for `agent`'s parameters.
        storage: The completed rollout to train on.
        advantages: Per-step advantages from `_gae`.
        targets: Per-step value targets from `_gae`.
        config: Hyperparameters for the update.
        edge_index: The instance's conflict-graph edge index.
        edge_attr: The instance's conflict-graph edge features.
        rng: Source of randomness for minibatch shuffling.
    """
    total = storage.rewards.shape[0]
    size = max(1, total // config.minibatches)
    chunk = size
    n_edges = int(edge_index.shape[1])
    if n_edges:
        chunk = max(1, min(size, config.max_minibatch_edges // n_edges))
    for _ in range(config.update_epochs):
        order = rng.permutation(total)
        for start in range(0, total, size):
            index = torch.as_tensor(order[start : start + size], device=storage.rewards.device)
            # Normalise over the whole minibatch, not each chunk, so chunking is invisible to
            # the update: what the optimiser sees must not depend on how much memory we had.
            batch_advantages = advantages[index]
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std() + 1e-8
            )
            optimiser.zero_grad()
            for offset in range(0, int(index.shape[0]), chunk):
                part = index[offset : offset + chunk]
                share = int(part.shape[0]) / int(index.shape[0])
                loss = _loss(
                    agent,
                    storage,
                    part,
                    batch_advantages[offset : offset + chunk],
                    targets[part],
                    config,
                    edge_index,
                    edge_attr,
                )
                (loss * share).backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimiser.step()


def _loss(  # noqa: PLR0913, PLR0917 - the clipped surrogate genuinely needs all of these
    agent: ActorCritic,
    storage: _Rollout,
    index: torch.Tensor,
    batch_advantages: torch.Tensor,
    batch_targets: torch.Tensor,
    config: PPOConfig,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """The clipped surrogate plus value and entropy terms, for one slice of a minibatch.

    Args:
        agent: The network being trained.
        storage: The completed rollout to train on.
        index: Which rollout steps this slice covers.
        batch_advantages: Advantages for `index`, already normalised over the whole minibatch.
        batch_targets: Value targets for `index`.
        config: Hyperparameters for the update.
        edge_index: The instance's conflict-graph edge index.
        edge_attr: The instance's conflict-graph edge features.

    Returns:
        The scalar loss, averaged over this slice.
    """
    batch = EncoderInput(
        nodes=storage.nodes[index],
        globals=storage.globals[index],
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    logits, values = agent(batch, storage.periods[index])
    distribution = masked_categorical(logits, storage.masks[index])
    log_probs = distribution.log_prob(storage.actions[index])  # type: ignore[no-untyped-call]

    ratio = (log_probs - storage.log_probs[index]).exp()
    policy_loss = torch.max(
        -batch_advantages * ratio,
        -batch_advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef),
    ).mean()
    value_loss = 0.5 * (values - batch_targets).pow(2).mean()
    entropy = distribution.entropy().mean()  # type: ignore[no-untyped-call]
    loss: torch.Tensor = (
        policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    )
    return loss
