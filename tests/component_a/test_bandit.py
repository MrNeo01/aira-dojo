"""Dependency-free checks for the FAS Component-A policy."""

from __future__ import annotations

import json
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dojo.components.component_a.bandit import (  # noqa: E402
    ARMS,
    PhaseAwareFRRUCBPolicy,
    budget_phase,
    incumbent_progress_reward,
    require_validation_only,
)
from dojo.core.solvers.utils.action_mask import (  # noqa: E402
    consecutive_debug_depth,
    is_debuggable,
)


class PhaseAndRewardTests(unittest.TestCase):
    def test_phase_boundaries(self) -> None:
        self.assertEqual(budget_phase(0.0), "early")
        self.assertEqual(budget_phase(math.nextafter(1.0 / 3.0, 0.0)), "early")
        self.assertEqual(budget_phase(1.0 / 3.0), "middle")
        self.assertEqual(budget_phase(math.nextafter(2.0 / 3.0, 0.0)), "middle")
        self.assertEqual(budget_phase(2.0 / 3.0), "late")
        self.assertEqual(budget_phase(1.0), "late")

    def test_direction_aware_incumbent_progress(self) -> None:
        self.assertAlmostEqual(
            incumbent_progress_reward(0.4, 0.7, lower_is_better=False, cost=2.0),
            0.15,
        )
        self.assertAlmostEqual(
            incumbent_progress_reward(0.7, 0.4, lower_is_better=True, cost=2.0),
            0.15,
        )
        self.assertEqual(
            incumbent_progress_reward(0.7, 0.4, lower_is_better=False), 0.0
        )
        self.assertEqual(
            incumbent_progress_reward(None, 0.4, lower_is_better=False), 0.0
        )
        self.assertEqual(
            incumbent_progress_reward(
                0.0, 0.1, lower_is_better=False, cost=0.0, epsilon=1e-3
            ),
            100.0,
        )
        with self.assertRaises(OverflowError):
            incumbent_progress_reward(-1e308, 1e308, lower_is_better=False, cost=1.0)

    def test_private_score_mode_is_rejected(self) -> None:
        require_validation_only(False)
        with self.assertRaises(ValueError):
            require_validation_only(True)

    def test_shared_debug_mask_uses_explicit_operator_ancestry(self) -> None:
        root = SimpleNamespace(id="root", operators_used=[], parents=[])
        draft = SimpleNamespace(
            id="draft",
            operators_used=["draft"],
            parents=[root],
            is_buggy=True,
            is_leaf=True,
        )
        debug_one = SimpleNamespace(
            id="debug-1",
            operators_used=["debug"],
            parents=[draft],
            is_buggy=True,
            is_leaf=True,
        )
        debug_two = SimpleNamespace(
            id="debug-2",
            operators_used=["debug"],
            parents=[debug_one],
            is_buggy=True,
            is_leaf=True,
        )
        self.assertEqual(consecutive_debug_depth(draft), 0)
        self.assertEqual(consecutive_debug_depth(debug_one), 1)
        self.assertEqual(consecutive_debug_depth(debug_two), 2)
        self.assertTrue(is_debuggable(draft, 2))
        self.assertTrue(is_debuggable(debug_one, 2))
        self.assertFalse(is_debuggable(debug_two, 2))


class FormulaTests(unittest.TestCase):
    def test_manuscript_fitness_rate_rank_and_ucb_equations(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(
            window_size=8, decay=0.5, exploration=0.2, epsilon=1e-12
        )
        events = (
            ("draft", 0.0),
            ("improve", 0.12),
            ("improve", 0.08),
            ("improve", 0.04),
            ("debug", 0.02),
            ("debug", 0.06),
        )
        for operator, reward in events:
            policy.observe("early", operator, reward)

        snapshot = policy.snapshot("early", ARMS)
        self.assertEqual(snapshot.window_events, 6)
        self.assertEqual(snapshot.counts, {"draft": 1, "debug": 2, "improve": 3})
        self.assertAlmostEqual(snapshot.fitness_rates["draft"], 0.0)
        self.assertAlmostEqual(snapshot.fitness_rates["debug"], 0.04)
        self.assertAlmostEqual(snapshot.fitness_rates["improve"], 0.08)
        self.assertEqual(snapshot.ranks, {"draft": 3.0, "debug": 2.0, "improve": 1.0})
        self.assertAlmostEqual(snapshot.credits["draft"], 0.0, places=10)
        self.assertAlmostEqual(snapshot.credits["debug"], 0.2, places=10)
        self.assertAlmostEqual(snapshot.credits["improve"], 0.8, places=10)
        self.assertAlmostEqual(snapshot.scores["draft"], 0.2789917668, places=9)
        self.assertAlmostEqual(snapshot.scores["debug"], 0.4277958237, places=9)
        self.assertAlmostEqual(snapshot.scores["improve"], 0.9972769702, places=9)
        self.assertEqual(policy.choose(0.1, ARMS).operator, "improve")

    def test_average_tie_ranks_and_zero_credit(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(exploration=0.0)
        for operator in ARMS:
            policy.observe("early", operator, 0.0)
        snapshot = policy.snapshot("early", ARMS)
        self.assertEqual(snapshot.ranks, {operator: 2.0 for operator in ARMS})
        self.assertEqual(snapshot.credits, {operator: 0.0 for operator in ARMS})
        self.assertEqual(snapshot.scores, {operator: 0.0 for operator in ARMS})

    def test_ranks_include_observed_arms_outside_current_mask(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(decay=0.5, exploration=0.0, epsilon=1e-12)
        policy.observe("early", "draft", 0.9)
        policy.observe("early", "debug", 0.8)
        policy.observe("early", "improve", 0.7)

        snapshot = policy.snapshot("early", ("draft", "improve"))
        self.assertEqual(
            snapshot.ranks,
            {"draft": 1.0, "debug": 2.0, "improve": 3.0},
        )
        self.assertAlmostEqual(snapshot.credits["draft"], 0.9 / 1.075)
        self.assertAlmostEqual(snapshot.credits["improve"], 0.175 / 1.075)
        self.assertNotIn("debug", snapshot.credits)
        self.assertNotIn("debug", snapshot.scores)

    def test_credit_denominator_includes_configured_epsilon(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(decay=0.5, exploration=0.0, epsilon=0.1)
        policy.observe("early", "draft", 1.0)
        snapshot = policy.snapshot("early", ("draft",))
        self.assertAlmostEqual(snapshot.credits["draft"], 1.0 / 1.1)

    def test_common_window_evicts_events_but_not_forced_once_memory(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(window_size=3, seed=4)
        for operator in ARMS:
            policy.observe("early", operator, 0.0)
        policy.observe("early", "debug", 0.1)
        self.assertEqual(
            policy.events("early"),
            (("debug", 0.0), ("improve", 0.0), ("debug", 0.1)),
        )
        policy.observe("early", "debug", 0.1)
        policy.observe("early", "debug", 0.1)
        self.assertEqual(policy.snapshot("early", ARMS).counts["draft"], 0)
        decision = policy.choose(0.1, ARMS)
        self.assertFalse(decision.forced)
        self.assertEqual(policy.pulled_once("early"), frozenset(ARMS))


class ExplorationAndStateTests(unittest.TestCase):
    def test_newly_feasible_arm_is_forced_once_per_phase(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(seed=9)
        first = policy.choose(0.1, ("draft",))
        self.assertTrue(first.forced)
        self.assertEqual(first.operator, "draft")
        policy.observe(first.phase, first.operator, 0.0)

        second = policy.choose(0.1, ("draft", "debug"))
        self.assertTrue(second.forced)
        self.assertEqual(second.operator, "debug")
        policy.observe(second.phase, second.operator, 0.0)
        self.assertFalse(policy.choose(0.1, ("draft", "debug")).forced)

        late = policy.choose(0.8, ("draft",))
        self.assertTrue(late.forced)
        self.assertEqual(late.operator, "draft")

    def test_action_mask_is_respected(self) -> None:
        policy = PhaseAwareFRRUCBPolicy(seed=2)
        for fraction in (0.1, 0.4, 0.8):
            for _ in range(8):
                decision = policy.choose(fraction, ("draft",))
                self.assertEqual(decision.operator, "draft")
                policy.observe(decision.phase, decision.operator, 1.0)

    def test_seeded_uniform_ties_are_reproducible(self) -> None:
        def sequence(seed: int) -> list[str]:
            policy = PhaseAwareFRRUCBPolicy(exploration=0.0, seed=seed)
            for operator in ARMS:
                policy.observe("early", operator, 0.0)
            return [policy.choose(0.1, ARMS).operator for _ in range(12)]

        self.assertEqual(sequence(17), sequence(17))
        self.assertNotEqual(sequence(17), sequence(18))

    def test_json_state_round_trip_preserves_rng_and_windows(self) -> None:
        original = PhaseAwareFRRUCBPolicy(window_size=4, seed=23)
        for operator, reward in zip(ARMS, (0.1, 0.0, 0.2)):
            original.observe("middle", operator, reward)
        state = json.loads(json.dumps(original.state_dict()))

        restored = PhaseAwareFRRUCBPolicy(window_size=4, seed=999)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        original_choices = [original.choose(0.5, ARMS).operator for _ in range(10)]
        restored_choices = [restored.choose(0.5, ARMS).operator for _ in range(10)]
        self.assertEqual(original_choices, restored_choices)


class BehavioralProofTests(unittest.TestCase):
    def test_phase_specific_policy_tracks_changing_best_arm(self) -> None:
        examples_dir = REPOSITORY_ROOT / "examples"
        if str(examples_dir) not in sys.path:
            sys.path.insert(0, str(examples_dir))
        from fas_component_a_poc import run_proof

        result = run_proof(budget=90, warm_start=3, seed=7)
        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["reward_improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()
