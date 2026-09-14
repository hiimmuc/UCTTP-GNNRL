"""Inference: run a trained policy on an environment and keep the best timetable it produces."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from horarium.agents._obs import edge_tensors, encoder_input, period_batch
from horarium.envs.base import EpisodeStats, TimetablingEnv
from horarium.models.masking import masked_categorical
from horarium.problem.cost import cost
from horarium.problem.solution import Solution
from horarium.solvers.rooms import RoomAssignmentError, reassign_rooms

__all__ = ["Attempt", "SolveResult", "polish", "rollout", "solve"]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One episode of inference."""

    stats: EpisodeStats
    solution: Solution
    greedy: bool
    seed: int


@dataclass(frozen=True, slots=True)
class SolveResult:
    """What a batch of inference attempts produced."""

    best: Attempt | None
    attempts: int
    feasible: int

    @property
    def feasibility_rate(self) -> float:
        """Fraction of attempts that produced a complete, hard-feasible timetable."""
        return self.feasible / self.attempts if self.attempts else 0.0


@torch.no_grad()
def rollout(
    env: TimetablingEnv,
    agent: torch.nn.Module,
    *,
    greedy: bool = True,
    seed: int = 0,
    device: torch.device | None = None,
) -> Attempt:
    """Play one episode. Greedy takes the highest-scoring legal action; otherwise it samples.

    A dead end ends the episode like any other terminal state and is reported as infeasible,
    rather than raising: an under-trained policy failing to finish is a result, not an error.

    Args:
        env: The environment to play the episode in.
        agent: The trained network to act with.
        greedy: Whether to always take the highest-scoring legal action, or sample.
        seed: Seed for the episode and, when sampling, the action draws.
        device: Device to run inference on. Defaults to the agent's own device.

    Returns:
        What the episode achieved and the timetable it produced.
    """
    where = device or next(agent.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    edge_index, edge_attr = edge_tensors(env, where)

    observation, _ = env.reset(seed=seed)
    stats: EpisodeStats | None = None
    while True:
        mask = torch.as_tensor(observation["action_mask"], device=where).bool().unsqueeze(0)
        if not bool(mask.any()):
            break
        batch = encoder_input(observation, edge_index, edge_attr, where)
        logits, _value = agent(batch, period_batch(observation, where))
        distribution = masked_categorical(logits, mask)
        if greedy:
            action = int(distribution.logits.argmax(dim=-1).item())
        else:
            probabilities = distribution.probs.squeeze(0).cpu()
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())

        observation, _reward, terminated, truncated, info = env.step(np.int64(action))
        if terminated or truncated:
            stats = info["episode_stats"]
            break

    if stats is None:  # the very first state had no legal action
        stats = env.episode_stats(dead_end=True)
    return Attempt(stats=stats, solution=env.solution.copy(), greedy=greedy, seed=seed)


def polish(env: TimetablingEnv, attempt: Attempt) -> Attempt:
    """Re-solve the attempt's rooms exactly, keeping every lecture in the period the policy chose.

    The policy decides the master problem -- which period each lecture goes in -- and the rooms
    are the subproblem. During an episode the env fills them with a fast greedy rule so each state
    stays cost-evaluable, but the objective a solve reports should be the one the two-stage
    method actually achieves, which means the exact stage-2 optimum. It is a few milliseconds per
    attempt and never makes a solution worse.

    Args:
        env: The environment the attempt came from, for its instance and formulation.
        attempt: The attempt to re-room.

    Returns:
        The attempt with optimal rooms and its cost restated, or unchanged if no assignment
        exists (which cannot happen for a solution the env already built, but is not worth
        crashing a sweep over).
    """
    try:
        improved = reassign_rooms(attempt.solution, env.formulation)
    except RoomAssignmentError:  # pragma: no cover - the env only ever builds roomable solutions
        return attempt
    breakdown = cost(env.instance, improved, env.formulation)
    stats = replace(
        attempt.stats, total_cost=float(breakdown.total), violations=breakdown.violations
    )
    return replace(attempt, stats=stats, solution=improved)


def solve(
    env: TimetablingEnv,
    agent: torch.nn.Module,
    *,
    restarts: int = 1,
    greedy: bool = False,
    seed: int = 0,
) -> SolveResult:
    """Run several attempts and keep the cheapest feasible timetable.

    Every feasible attempt is passed through `polish` first, so attempts are ranked by the cost
    the two-stage method really delivers rather than by the env's placeholder rooming.

    The greedy rollout is deterministic, so a fully greedy solve does one attempt however many
    restarts are asked for -- repeating it would report one answer several times over. Otherwise
    the first attempt is greedy and the rest sample, so restarts explore. The device is taken
    from the agent's own parameters.

    Args:
        env: The environment to solve.
        agent: The trained network to act with.
        restarts: Number of attempts, when not `greedy`.
        greedy: Whether to always take the highest-scoring legal action, or sample.
        seed: Seed for the first attempt; later attempts use `seed + index`.

    Returns:
        The cheapest feasible attempt found, and how many of the attempts were feasible.
    """
    attempts = 1 if greedy else max(1, restarts)
    best: Attempt | None = None
    feasible = 0
    for index in range(attempts):
        attempt = rollout(env, agent, greedy=greedy or index == 0, seed=seed + index)
        if not attempt.stats.feasible:
            continue
        feasible += 1
        improved = polish(env, attempt)
        if best is None or improved.stats.total_cost < best.stats.total_cost:
            best = improved
    return SolveResult(best=best, attempts=attempts, feasible=feasible)
