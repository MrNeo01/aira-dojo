#!/usr/bin/env python3
"""Dependency-free behavioral proof for FAS Component A.

This tiny environment changes which operator pays off in each budget phase. It
tests whether Component A follows that change and compares its accumulated
synthetic reward with a phase-agnostic round-robin allocation. It is a software
sanity check, not evidence about MLE-bench performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dojo.components.component_a.bandit import (  # noqa: E402
    ARMS,
    PHASES,
    PhaseAwareFRRUCBPolicy,
    budget_phase,
)

BEST_BY_PHASE = {
    "early": "draft",
    "middle": "debug",
    "late": "improve",
}


def synthetic_reward(phase: str, operator: str) -> float:
    return 1.0 if operator == BEST_BY_PHASE[phase] else 0.0


def run_component_a(*, budget: int, warm_start: int, seed: int) -> dict[str, Any]:
    policy = PhaseAwareFRRUCBPolicy(seed=seed)
    allocations = {phase: Counter() for phase in PHASES}
    total_reward = 0.0

    for evaluation in range(budget):
        fraction = evaluation / budget
        phase = budget_phase(fraction)
        if evaluation < warm_start:
            # Common charged initialization: deliberately not observed by policy.
            operator = "draft"
        else:
            decision = policy.choose(fraction, ARMS)
            operator = decision.operator
        reward = synthetic_reward(phase, operator)
        allocations[phase][operator] += 1
        total_reward += reward
        if evaluation >= warm_start:
            policy.observe(phase, operator, reward)

    return {
        "method": "fas_component_a",
        "budget": budget,
        "warm_start": warm_start,
        "reward": total_reward,
        "allocations": {
            phase: {operator: allocations[phase][operator] for operator in ARMS}
            for phase in PHASES
        },
    }


def run_round_robin(*, budget: int, warm_start: int) -> dict[str, Any]:
    allocations = {phase: Counter() for phase in PHASES}
    total_reward = 0.0
    cursor = 0
    for evaluation in range(budget):
        phase = budget_phase(evaluation / budget)
        if evaluation < warm_start:
            operator = "draft"
        else:
            operator = ARMS[cursor % len(ARMS)]
            cursor += 1
        allocations[phase][operator] += 1
        total_reward += synthetic_reward(phase, operator)
    return {
        "method": "round_robin",
        "budget": budget,
        "warm_start": warm_start,
        "reward": total_reward,
        "allocations": {
            phase: {operator: allocations[phase][operator] for operator in ARMS}
            for phase in PHASES
        },
    }


def run_proof(
    *, budget: int = 90, warm_start: int = 3, seed: int = 7
) -> dict[str, Any]:
    if budget < 18:
        raise ValueError("budget must be at least 18")
    if not 0 <= warm_start < budget // 3:
        raise ValueError("warm_start must fit inside the early phase")
    adaptive = run_component_a(budget=budget, warm_start=warm_start, seed=seed)
    control = run_round_robin(budget=budget, warm_start=warm_start)

    for phase, best_operator in BEST_BY_PHASE.items():
        phase_counts = adaptive["allocations"][phase]
        best_count = phase_counts[best_operator]
        if best_count <= max(
            count
            for operator, count in phase_counts.items()
            if operator != best_operator
        ):
            raise AssertionError(f"Component A did not adapt in phase {phase}")
    if adaptive["reward"] <= control["reward"]:
        raise AssertionError("Component A did not beat the round-robin control")

    return {
        "status": "passed",
        "task": "phase_changing_operator_rewards",
        "best_operator_by_phase": BEST_BY_PHASE,
        "component_a": adaptive,
        "control": control,
        "reward_improvement": adaptive["reward"] - control["reward"],
        "warning": "Behavioral software proof only; not an MLE-bench result.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--warm-start", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_proof(
                budget=args.budget,
                warm_start=args.warm_start,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
