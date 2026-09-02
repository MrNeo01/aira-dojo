"""AIRA-dojo adapter for FAS Component A.

The adapter subclasses Greedy and preserves its prompts, operator
implementations, execution, analysis, validation, Journal sequence, and
stopping rule.  Its ``step`` follows that flow while replacing the
operator/parent decision and adding post-evaluation policy observation.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

from dojo.components.component_a.bandit import (
    PhaseAwareFRRUCBPolicy,
    budget_phase,
    incumbent_progress_reward,
    require_validation_only,
)
from dojo.core.solvers.utils.action_mask import eligible_debug_nodes
from dojo.core.solvers.utils.response import extract_code
from dojo.solvers.greedy.greedy import Greedy

CHECKPOINT_VERSION = 1
PARENT_RNG_XOR = 0xA17A


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


class ComponentASolver(Greedy):
    """Greedy search mechanics with the FAS Component-A operator policy."""

    search_method_name = "FAS_Component_A"

    def __init__(self, cfg: Any, task_info: Mapping[str, Any]) -> None:
        require_validation_only(bool(cfg.use_test_score))
        super().__init__(cfg, task_info)
        self.component_a_policy = PhaseAwareFRRUCBPolicy(
            window_size=cfg.component_a_window_size,
            decay=cfg.component_a_decay,
            exploration=cfg.component_a_exploration,
            epsilon=cfg.component_a_epsilon,
            phase_boundaries=cfg.component_a_phase_boundaries,
            seed=cfg.component_a_seed,
        )
        # Keep parent sampling independent of policy tie-breaking consumption.
        self.component_a_parent_rng = random.Random(
            int(cfg.component_a_seed) ^ PARENT_RNG_XOR
        )
        self.last_policy_record: dict[str, Any] | None = None

    def create_root_node(self) -> None:
        """Create a root for a fresh run or restore it without duplicating it."""

        if not self.journal.nodes:
            super().create_root_node()
            return
        root = self.journal.nodes[0]
        if not self.journal.is_root_node(root):
            raise RuntimeError("checkpoint Journal does not start with an AIRA root")
        if int(self.state.current_step) != len(self.journal.nodes):
            raise RuntimeError("checkpoint state and Journal lengths are inconsistent")
        self.root_node = root

    @staticmethod
    def _metric_value(node: Any | None) -> float | None:
        if node is None or node.metric is None or node.metric.value is None:
            return None
        value = float(node.metric.value)
        return value if math.isfinite(value) else None

    def _eligible_nodes(self) -> dict[str, list[Any]]:
        debuggable = eligible_debug_nodes(
            self.journal.buggy_nodes, self.cfg.max_debug_depth
        )
        return {
            "draft": [],
            "debug": debuggable,
            "improve": list(self.journal.good_nodes),
        }

    def _budget_state(self) -> tuple[int, int, float]:
        evaluation_limit = max(1, int(self.cfg.step_limit) - 1)
        # The Journal includes one synthetic root which is not an evaluation.
        evaluations_used = max(0, len(self.journal.nodes) - 1)
        fraction = min(1.0, evaluations_used / evaluation_limit)
        return evaluations_used, evaluation_limit, fraction

    def _choose_action(self) -> dict[str, Any]:
        eligible_nodes = self._eligible_nodes()
        eligible_operators = tuple(
            operator
            for operator in ("draft", "debug", "improve")
            if operator == "draft" or eligible_nodes[operator]
        )
        evaluations_used, evaluation_limit, budget_fraction = self._budget_state()
        phase = budget_phase(budget_fraction, self.component_a_policy.phase_boundaries)
        warm_start = len(self.journal.draft_nodes) < self.cfg.num_drafts

        if warm_start:
            operator = "draft"
            snapshot = self.component_a_policy.snapshot(phase, eligible_operators)
            bandit_record: dict[str, Any] = {
                "operator": operator,
                "phase": phase,
                "forced": False,
                "forced_candidates": [],
                **snapshot.as_dict(),
            }
            selection_reason = "common_draft_warm_start"
        else:
            decision = self.component_a_policy.choose(
                budget_fraction, eligible_operators
            )
            operator = decision.operator
            phase = decision.phase
            bandit_record = decision.as_dict()
            selection_reason = "forced_once_per_phase" if decision.forced else "frr_ucb"

        parent = None
        if operator == "debug":
            parent = self.component_a_parent_rng.choice(eligible_nodes["debug"])
        elif operator == "improve":
            parent = self.journal.get_best_node()
            if parent is None:
                raise RuntimeError("Improve was selected without a valid parent")

        return {
            "operator": operator,
            "parent": parent,
            "parent_ids": [] if parent is None else [str(parent.id)],
            "eligible_operators": list(eligible_operators),
            "evaluations_used": evaluations_used,
            "evaluation_limit": evaluation_limit,
            "budget_fraction": budget_fraction,
            "phase": phase,
            "warm_start": warm_start,
            "selection_reason": selection_reason,
            "bandit": bandit_record,
        }

    def save_checkpoint(self) -> None:
        """Save stock solver data and a small Component-A state sidecar."""

        super().save_checkpoint()
        checkpoint_path = Path(self.cfg.checkpoint_path)
        payload = {
            "version": CHECKPOINT_VERSION,
            "policy": self.component_a_policy.state_dict(),
            "parent_rng_state": _json_safe(self.component_a_parent_rng.getstate()),
        }
        (checkpoint_path / "component_a_policy.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def load_checkpoint(self) -> None:
        """Restore Component-A state when the matching sidecar exists."""

        super().load_checkpoint()
        policy_path = Path(self.cfg.checkpoint_path) / "component_a_policy.json"
        has_search_state = bool(self.journal.nodes) or self.state.current_step > 0
        if not policy_path.exists():
            if has_search_state:
                raise RuntimeError(
                    "Component-A search checkpoint exists without its policy sidecar"
                )
            return
        if not has_search_state:
            raise RuntimeError(
                "Component-A policy sidecar exists without a search checkpoint"
            )
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("unsupported Component-A checkpoint version")
        self.component_a_policy.load_state_dict(payload["policy"])
        self.component_a_parent_rng.setstate(_nested_tuple(payload["parent_rng_state"]))

    def step(self, task: Any, state: Any):
        """Select, execute, evaluate, and learn from one feasible operator."""

        self.logger.info(f"Step {self.state.current_step}: Starting FAS-A iteration")
        action_started = time.monotonic()

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        incumbent_before_node = self.journal.get_best_node()
        incumbent_before = self._metric_value(incumbent_before_node)
        action = self._choose_action()
        operator = action["operator"]
        parent = action["parent"]

        if operator == "draft":
            result_node = self._draft()
        elif operator == "debug":
            result_node = self._debug(parent)
        elif operator == "improve":
            result_node = self._improve(parent)
        else:  # The feasibility builder and bandit both reject unknown arms.
            raise RuntimeError(f"unsupported Component-A operator: {operator!r}")

        state, eval_result = task.step_task(state, extract_code(result_node.code))
        self.parse_eval_result(node=result_node, eval_result=eval_result)
        self.journal.append(result_node)

        incumbent_after_node = self.journal.get_best_node()
        incumbent_after = self._metric_value(incumbent_after_node)
        reward = incumbent_progress_reward(
            incumbent_before,
            incumbent_after,
            lower_is_better=bool(self.lower_is_better),
            cost=1.0,
            epsilon=self.cfg.component_a_epsilon,
        )
        learned = not action["warm_start"]
        if learned:
            self.component_a_policy.observe(action["phase"], operator, reward)

        record = {
            "policy_name": self.component_a_policy.name,
            "step": self.state.current_step,
            "child_id": str(result_node.id),
            "child_validation_metric": self._metric_value(result_node),
            "child_is_buggy": bool(result_node.is_buggy),
            "operator": operator,
            "parent_ids": action["parent_ids"],
            "eligible_operators": action["eligible_operators"],
            "phase": action["phase"],
            "budget_fraction": action["budget_fraction"],
            "evaluations_used_before_action": action["evaluations_used"],
            "evaluation_limit": action["evaluation_limit"],
            "selection_reason": action["selection_reason"],
            "warm_start": action["warm_start"],
            "learned_from_transition": learned,
            "incumbent_before": incumbent_before,
            "incumbent_after": incumbent_after,
            "reward": reward,
            "reward_cost": 1.0,
            "reward_cost_unit": "candidate_evaluation",
            "action_wall_time_secs": time.monotonic() - action_started,
            "bandit": action["bandit"],
        }
        self.last_policy_record = record
        self.logger.log(record, "POLICY", step=self.state.current_step)

        best_step = 0 if incumbent_after_node is None else incumbent_after_node.step
        self.logger.log(
            self.journal.get_node_data(self.state.current_step)
            | {"current_best_node": best_step},
            "JOURNAL",
            step=self.state.current_step,
        )
        self.logger.log(
            self.state.state_dict(),
            "STATE",
            step=self.state.current_step,
        )
        self.logger.info(f"Step {self.state.current_step}: FAS-A iteration complete")
        return state, eval_result
