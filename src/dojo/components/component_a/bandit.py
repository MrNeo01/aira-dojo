"""Manuscript-equation online bandit for FAS Component A.

The policy is deliberately dependency-free.  It maintains one common sliding
action/reward window per budget phase and selects only from the feasibility mask
provided by the solver.  No offline training or previous-run log is consumed.

The equations implemented here follow ``ICLR_paper/main.tex`` rather than the
older educational demo in ``basics/adaptive_search_methods_demo.py``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ARMS = ("draft", "debug", "improve")
PHASES = ("early", "middle", "late")
DEFAULT_PHASE_BOUNDARIES = (1.0 / 3.0, 2.0 / 3.0)
STATE_VERSION = 1


def _validate_phase_boundaries(boundaries: Sequence[float]) -> tuple[float, float]:
    if len(boundaries) != 2:
        raise ValueError("phase_boundaries must contain exactly two values")
    early_end, middle_end = (float(value) for value in boundaries)
    if not (0.0 < early_end < middle_end < 1.0):
        raise ValueError("phase_boundaries must satisfy 0 < early < middle < 1")
    return early_end, middle_end


def budget_phase(
    budget_fraction: float,
    boundaries: Sequence[float] = DEFAULT_PHASE_BOUNDARIES,
) -> str:
    """Map consumed primary-budget fraction to early, middle, or late."""

    fraction = float(budget_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("budget_fraction must be finite and in [0, 1]")
    early_end, middle_end = _validate_phase_boundaries(boundaries)
    if fraction < early_end:
        return "early"
    if fraction < middle_end:
        return "middle"
    return "late"


def require_validation_only(use_test_score: bool) -> None:
    """Reject the AIRA mode that replaces validation fitness with test score."""

    if use_test_score:
        raise ValueError(
            "FAS Component A requires use_test_score=False because its online "
            "reward must use validation fitness, never private test outcomes."
        )


def incumbent_progress_reward(
    incumbent_before: float | None,
    incumbent_after: float | None,
    *,
    lower_is_better: bool,
    cost: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """Return non-negative, direction-corrected incumbent progress per cost.

    A missing or non-finite incumbent yields zero.  In particular, the first
    valid solution is not assigned an arbitrary task-dependent improvement over
    an invented floor.
    """

    denominator_epsilon = float(epsilon)
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    action_cost = float(cost)
    if not math.isfinite(action_cost) or action_cost < 0.0:
        raise ValueError("cost must be finite and non-negative")
    if incumbent_before is None or incumbent_after is None:
        return 0.0
    before = float(incumbent_before)
    after = float(incumbent_after)
    if not math.isfinite(before) or not math.isfinite(after):
        return 0.0
    signed_progress = before - after if lower_is_better else after - before
    if not math.isfinite(signed_progress):
        raise OverflowError("incumbent difference exceeds finite float range")
    reward = max(0.0, signed_progress) / max(action_cost, denominator_epsilon)
    if not math.isfinite(reward):
        raise OverflowError("cost-normalized incumbent reward is not finite")
    return reward


@dataclass(frozen=True)
class BanditSnapshot:
    """Window statistics and scores for one decision-time action mask."""

    phase: str
    window_events: int
    counts: Mapping[str, int]
    fitness_rates: Mapping[str, float]
    ranks: Mapping[str, float]
    credits: Mapping[str, float]
    scores: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "window_events": self.window_events,
            "counts": dict(self.counts),
            "fitness_rates": dict(self.fitness_rates),
            "ranks": dict(self.ranks),
            "credits": dict(self.credits),
            "scores": dict(self.scores),
        }


@dataclass(frozen=True)
class BanditDecision:
    """Selected feasible arm plus the evidence used for the selection."""

    operator: str
    phase: str
    forced: bool
    forced_candidates: tuple[str, ...]
    snapshot: BanditSnapshot

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "phase": self.phase,
            "forced": self.forced,
            "forced_candidates": list(self.forced_candidates),
            **self.snapshot.as_dict(),
        }


def _average_descending_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Return one-based average ranks, with rank one assigned to the largest."""

    ranks: dict[str, float] = {}
    for operator, value in values.items():
        greater = sum(other_value > value for other_value in values.values())
        tied = sum(other_value == value for other_value in values.values())
        ranks[operator] = 1.0 + greater + (tied - 1.0) / 2.0
    return ranks


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class PhaseAwareFRRUCBPolicy:
    """Phase-aware Fitness-Rate-Rank credit with sliding-window UCB.

    ``observe`` must be called only for completed adaptive actions.  Common
    warm-start actions are intentionally excluded from the statistics.
    """

    name = "fas_component_a_frr_ucb"

    def __init__(
        self,
        *,
        window_size: int = 8,
        decay: float = 0.5,
        exploration: float = 0.25,
        epsilon: float = 1e-12,
        phase_boundaries: Sequence[float] = DEFAULT_PHASE_BOUNDARIES,
        seed: int = 0,
        arms: Sequence[str] = ARMS,
    ) -> None:
        if int(window_size) != window_size or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(float(decay)) or not 0.0 <= float(decay) <= 1.0:
            raise ValueError("decay must be finite and in [0, 1]")
        if not math.isfinite(float(exploration)) or float(exploration) < 0.0:
            raise ValueError("exploration must be finite and non-negative")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        configured_arms = tuple(str(arm) for arm in arms)
        if not configured_arms or len(set(configured_arms)) != len(configured_arms):
            raise ValueError("arms must be non-empty and unique")

        self.window_size = int(window_size)
        self.decay = float(decay)
        self.exploration = float(exploration)
        self.epsilon = float(epsilon)
        self.phase_boundaries = _validate_phase_boundaries(phase_boundaries)
        self.arms = configured_arms
        self._events: dict[str, deque[tuple[str, float]]] = {
            phase: deque(maxlen=self.window_size) for phase in PHASES
        }
        self._pulled_once: dict[str, set[str]] = {phase: set() for phase in PHASES}
        self._rng = random.Random(int(seed))

    def _eligible(self, eligible_operators: Sequence[str]) -> tuple[str, ...]:
        requested = tuple(str(operator) for operator in eligible_operators)
        if len(set(requested)) != len(requested):
            raise ValueError("eligible_operators must not contain duplicates")
        unknown = set(requested) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown eligible operators: {sorted(unknown)}")
        eligible_set = set(requested)
        eligible = tuple(operator for operator in self.arms if operator in eligible_set)
        if not eligible:
            raise ValueError("at least one operator must be eligible")
        return eligible

    def snapshot(self, phase: str, eligible_operators: Sequence[str]) -> BanditSnapshot:
        """Compute the current paper-defined statistics without changing state."""

        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        eligible = self._eligible(eligible_operators)
        events = self._events[phase]
        # Rates/ranks are phase-window properties of every observed arm.  The
        # current feasibility mask is applied only to credit normalization and
        # selection, as in the manuscript.
        counts = {operator: 0 for operator in self.arms}
        reward_sums = {operator: 0.0 for operator in self.arms}
        for operator, reward in events:
            counts[operator] += 1
            reward_sums[operator] += reward

        fitness_rates = {
            operator: reward_sums[operator] / max(1, counts[operator])
            for operator in self.arms
        }
        rank_operators = tuple(
            operator
            for operator in self.arms
            if counts[operator] > 0 or operator in eligible
        )
        ranks = _average_descending_ranks(
            {operator: fitness_rates[operator] for operator in rank_operators}
        )
        decayed = {
            operator: (self.decay ** (ranks[operator] - 1.0)) * fitness_rates[operator]
            for operator in eligible
        }
        credit_mass = sum(decayed.values())
        if credit_mass <= 0.0:
            credits = {operator: 0.0 for operator in eligible}
        else:
            denominator = credit_mass + self.epsilon
            credits = {
                operator: decayed[operator] / denominator for operator in eligible
            }

        window_events = len(events)
        scores = {
            operator: credits[operator]
            + self.exploration
            * math.sqrt(2.0 * math.log(1.0 + window_events) / (1.0 + counts[operator]))
            for operator in eligible
        }
        return BanditSnapshot(
            phase=phase,
            window_events=window_events,
            counts=counts,
            fitness_rates=fitness_rates,
            ranks=ranks,
            credits=credits,
            scores=scores,
        )

    def choose(
        self, budget_fraction: float, eligible_operators: Sequence[str]
    ) -> BanditDecision:
        """Select one feasible operator using forced exploration then FRR-UCB."""

        phase = budget_phase(budget_fraction, self.phase_boundaries)
        eligible = self._eligible(eligible_operators)
        snapshot = self.snapshot(phase, eligible)
        forced_candidates = tuple(
            operator
            for operator in eligible
            if operator not in self._pulled_once[phase]
        )
        if forced_candidates:
            operator = self._rng.choice(forced_candidates)
            forced = True
        else:
            best_score = max(snapshot.scores.values())
            tied = tuple(
                operator
                for operator in eligible
                if snapshot.scores[operator] == best_score
            )
            operator = self._rng.choice(tied)
            forced = False
        return BanditDecision(
            operator=operator,
            phase=phase,
            forced=forced,
            forced_candidates=forced_candidates,
            snapshot=snapshot,
        )

    def observe(self, phase: str, operator: str, reward: float) -> None:
        """Record one evaluated adaptive action in its decision-time phase."""

        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        if operator not in self.arms:
            raise ValueError(f"unknown operator: {operator!r}")
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward) or numeric_reward < 0.0:
            raise ValueError("reward must be finite and non-negative")
        self._events[phase].append((operator, numeric_reward))
        self._pulled_once[phase].add(operator)

    def pulled_once(self, phase: str) -> frozenset[str]:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        return frozenset(self._pulled_once[phase])

    def events(self, phase: str) -> tuple[tuple[str, float], ...]:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        return tuple(self._events[phase])

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy and tie-RNG checkpoint."""

        return {
            "version": STATE_VERSION,
            "window_size": self.window_size,
            "decay": self.decay,
            "exploration": self.exploration,
            "epsilon": self.epsilon,
            "phase_boundaries": list(self.phase_boundaries),
            "arms": list(self.arms),
            "events": {
                phase: [
                    {"operator": operator, "reward": reward}
                    for operator, reward in self._events[phase]
                ]
                for phase in PHASES
            },
            "pulled_once": {
                phase: [
                    operator
                    for operator in self.arms
                    if operator in self._pulled_once[phase]
                ]
                for phase in PHASES
            },
            "rng_state": _json_safe(self._rng.getstate()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a matching policy checkpoint, rejecting config drift."""

        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported Component-A policy state version")
        expected = {
            "window_size": self.window_size,
            "decay": self.decay,
            "exploration": self.exploration,
            "epsilon": self.epsilon,
            "phase_boundaries": list(self.phase_boundaries),
            "arms": list(self.arms),
        }
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise ValueError(f"policy checkpoint does not match {key}")

        restored_events: dict[str, deque[tuple[str, float]]] = {}
        restored_pulled: dict[str, set[str]] = {}
        raw_events = state.get("events", {})
        raw_pulled = state.get("pulled_once", {})
        for phase in PHASES:
            phase_events = list(raw_events.get(phase, []))
            if len(phase_events) > self.window_size:
                raise ValueError("policy checkpoint contains an oversized window")
            event_window: deque[tuple[str, float]] = deque(maxlen=self.window_size)
            for event in phase_events:
                operator = str(event["operator"])
                reward = float(event["reward"])
                if (
                    operator not in self.arms
                    or not math.isfinite(reward)
                    or reward < 0.0
                ):
                    raise ValueError("invalid event in policy checkpoint")
                event_window.append((operator, reward))
            seen = {str(operator) for operator in raw_pulled.get(phase, [])}
            if not seen <= set(self.arms):
                raise ValueError("invalid pulled_once arm in policy checkpoint")
            restored_events[phase] = event_window
            restored_pulled[phase] = seen

        self._events = restored_events
        self._pulled_once = restored_pulled
        self._rng.setstate(_nested_tuple(state["rng_state"]))
