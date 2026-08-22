#!/usr/bin/env python3
"""Policy hooks for experimenting with adaptive search in AIRA-dojo.

This file is deliberately a scaffold, not the Paper-1 method.  It provides:

* a leakage-safe view of Dojo nodes;
* separate node-selector and operator-policy interfaces;
* fixed, random, and round-robin comparison policies; and
* a factory for a ``Greedy`` subclass whose ``step`` method dispatches an
  explicit policy decision through the existing Dojo operators/evaluator.

The policy classes and their self-test use only the Python standard library.
Creating the Dojo adapter requires an installed aira-dojo environment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

OperatorName = str


@dataclass(frozen=True)
class OnlineNodeView:
    """Information that an online policy is allowed to observe.

    ``metric_info`` is intentionally absent.  On MLE-bench it contains the
    private test score, which must never be available to an online policy.
    """

    node_id: str
    step: int
    validation_metric: float | None
    maximize: bool
    is_buggy: bool
    is_leaf: bool
    depth: int
    debug_depth: int
    operators_used: tuple[str, ...]
    execution_time: float | None
    parent_ids: tuple[str, ...] = ()
    code_text: str = ""
    public_error_class: str | None = None

    @property
    def utility(self) -> float:
        """Return a higher-is-better representation of validation fitness."""

        if self.validation_metric is None:
            return -math.inf
        return self.validation_metric if self.maximize else -self.validation_metric


@dataclass(frozen=True)
class SearchContext:
    """Leakage-safe state passed to an operator policy."""

    step: int
    step_limit: int
    evaluations_used: int
    evaluation_limit: int
    remaining_evaluations: int
    elapsed_secs: float
    time_limit_secs: float
    budget_fraction: float
    draft_count: int
    good_count: int
    buggy_count: int
    eligible_operators: tuple[OperatorName, ...]
    nodes: tuple[OnlineNodeView, ...]


@dataclass(frozen=True)
class SearchDecision:
    """One factorized search decision."""

    operator: OperatorName
    parent_ids: tuple[str, ...]
    policy_name: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyTransition:
    """Observable result sent back to the operator policy after evaluation."""

    step: int
    decision: SearchDecision
    child_id: str
    child_validation_metric: float | None
    child_is_buggy: bool
    incumbent_before: float | None
    incumbent_after: float | None
    incumbent_delta: float | None
    execution_time: float | None


class NodeSelector(Protocol):
    """Select parent nodes after an operator has been chosen."""

    name: str

    def select(
        self,
        context: SearchContext,
        candidates: Sequence[OnlineNodeView],
        count: int,
        rng: random.Random,
    ) -> tuple[OnlineNodeView, ...]: ...


class OperatorPolicy(Protocol):
    """Choose among the operators that are legal in the current state."""

    name: str

    def choose(self, context: SearchContext, rng: random.Random) -> OperatorName: ...

    def observe(self, transition: PolicyTransition) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class BestValidationSelector:
    """Reproduce Greedy's best-validation parent choice."""

    name = "best_validation"

    def select(
        self,
        context: SearchContext,
        candidates: Sequence[OnlineNodeView],
        count: int,
        rng: random.Random,
    ) -> tuple[OnlineNodeView, ...]:
        del context, rng
        if count < 1:
            return ()
        if len(candidates) < count:
            raise ValueError(f"Need {count} candidates, received {len(candidates)}")
        ordered = sorted(
            candidates, key=lambda node: (node.utility, -node.step), reverse=True
        )
        return tuple(ordered[:count])


class RandomNodeSelector:
    """A cheap node-selection control."""

    name = "random_node"

    def select(
        self,
        context: SearchContext,
        candidates: Sequence[OnlineNodeView],
        count: int,
        rng: random.Random,
    ) -> tuple[OnlineNodeView, ...]:
        del context
        if len(candidates) < count:
            raise ValueError(f"Need {count} candidates, received {len(candidates)}")
        return tuple(rng.sample(list(candidates), k=count))


class StockGreedyNodeSelector:
    """Match stock Greedy: random Debug parent, best Improve parent.

    The adapter supplies a homogeneous candidate set after choosing an
    operator, so buggy candidates identify the Debug case.
    """

    name = "stock_greedy_node"

    def select(
        self,
        context: SearchContext,
        candidates: Sequence[OnlineNodeView],
        count: int,
        rng: random.Random,
    ) -> tuple[OnlineNodeView, ...]:
        if len(candidates) < count:
            raise ValueError(f"Need {count} candidates, received {len(candidates)}")
        if candidates and all(node.is_buggy for node in candidates):
            return tuple(rng.sample(list(candidates), k=count))
        return BestValidationSelector().select(context, candidates, count, rng)


class FixedGreedyOperatorPolicy:
    """Stock-like rule: debug probabilistically, otherwise improve, else draft."""

    name = "fixed_greedy"

    def __init__(self, debug_probability: float = 1.0) -> None:
        if not 0.0 <= debug_probability <= 1.0:
            raise ValueError("debug_probability must be in [0, 1]")
        self.debug_probability = debug_probability

    def choose(self, context: SearchContext, rng: random.Random) -> OperatorName:
        eligible = set(context.eligible_operators)
        if "debug" in eligible and rng.random() < self.debug_probability:
            return "debug"
        if "improve" in eligible:
            return "improve"
        if "draft" in eligible:
            return "draft"
        raise RuntimeError("No eligible operator")

    def observe(self, transition: PolicyTransition) -> None:
        del transition

    def state_dict(self) -> dict[str, Any]:
        return {"debug_probability": self.debug_probability}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.debug_probability = float(state["debug_probability"])


class RandomOperatorPolicy:
    """Uniformly sample from the feasible action mask."""

    name = "random_operator"

    def choose(self, context: SearchContext, rng: random.Random) -> OperatorName:
        if not context.eligible_operators:
            raise RuntimeError("No eligible operator")
        return rng.choice(context.eligible_operators)

    def observe(self, transition: PolicyTransition) -> None:
        del transition

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        del state


class RoundRobinOperatorPolicy:
    """Cycle through operators while respecting state-dependent feasibility."""

    name = "round_robin_operator"

    def __init__(
        self, order: Sequence[OperatorName] = ("draft", "improve", "debug")
    ) -> None:
        if not order:
            raise ValueError("order must not be empty")
        self.order = tuple(order)
        self.cursor = 0

    def choose(self, context: SearchContext, rng: random.Random) -> OperatorName:
        del rng
        eligible = set(context.eligible_operators)
        for offset in range(len(self.order)):
            index = (self.cursor + offset) % len(self.order)
            operator = self.order[index]
            if operator in eligible:
                self.cursor = (index + 1) % len(self.order)
                return operator
        raise RuntimeError(f"No configured operator is eligible: {sorted(eligible)}")

    def observe(self, transition: PolicyTransition) -> None:
        del transition

    def state_dict(self) -> dict[str, Any]:
        return {"order": list(self.order), "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.order = tuple(str(item) for item in state["order"])
        self.cursor = int(state["cursor"])


def _depth(node: Any, active: set[str] | None = None) -> int:
    """Compute maximum parent depth for a tree or crossover DAG."""

    parents = list(getattr(node, "parents", None) or [])
    if not parents:
        return 0
    active = set() if active is None else set(active)
    node_id = str(getattr(node, "id", id(node)))
    if node_id in active:
        raise ValueError("Cycle detected in the search graph")
    active.add(node_id)
    return 1 + max(_depth(parent, active) for parent in parents)


def online_view(node: Any) -> OnlineNodeView:
    """Create the explicit online allowlist from a Dojo ``Node``."""

    metric = getattr(node, "metric", None)
    value = getattr(metric, "value", None)
    maximize = getattr(metric, "maximize", True)
    exit_code = getattr(node, "exit_code", None)
    if exit_code not in (None, 0):
        error_class = f"execution_exit_{exit_code}"
    elif bool(node.is_buggy):
        error_class = "invalid_or_unscored"
    else:
        error_class = None
    return OnlineNodeView(
        node_id=str(node.id),
        step=int(node.step),
        validation_metric=None if value is None else float(value),
        maximize=True if maximize is None else bool(maximize),
        is_buggy=bool(node.is_buggy),
        is_leaf=bool(node.is_leaf),
        depth=_depth(node),
        debug_depth=int(node.debug_depth),
        operators_used=tuple(str(name) for name in (node.operators_used or [])),
        execution_time=None if node.exec_time is None else float(node.exec_time),
        parent_ids=tuple(str(parent.id) for parent in (node.parents or [])),
        code_text=str(node.code or ""),
        public_error_class=error_class,
    )


def _nested_tuple(value: Any) -> Any:
    """Restore tuple structure after a JSON round trip."""

    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def require_leakage_safe_config(cfg: Any) -> None:
    """Reject the Dojo mode that copies private test score into the metric."""

    if bool(getattr(cfg, "use_test_score", False)):
        raise ValueError(
            "Adaptive online policies require cfg.use_test_score=False; "
            "otherwise node.metric.value may contain the private test score."
        )


def make_policy_driven_greedy_class() -> type:
    """Return a real Dojo ``Greedy`` subclass with factorized policy hooks.

    Imports are intentionally lazy so ``--self-test`` works before the full
    project environment is installed.
    """

    try:
        # Load the registry first to avoid exposing this checkout's circular
        # config/solver import when a concrete solver is imported directly.
        from dojo.config_dataclasses.solver import SOLVER_MAP
        from dojo.core.solvers.utils.response import extract_code
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The Dojo adapter requires the aira-dojo environment. Run from the "
            "repository after `pip install -e .` or use the provided conda environment. "
            f"First missing import: {error.name!r}."
        ) from error

    Greedy = SOLVER_MAP["GreedySolverConfig"]

    class PolicyDrivenGreedy(Greedy):
        """Greedy mechanics with explicit node- and operator-policy decisions.

        This first scaffold supports the three operators already initialized by
        ``Greedy``: draft, improve, and debug.  Crossover should be added only
        after every comparison solver has the same operator and prompt.
        """

        def __init__(
            self,
            cfg: Any,
            task_info: Mapping[str, Any],
            *,
            operator_policy: OperatorPolicy | None = None,
            node_selector: NodeSelector | None = None,
            policy_seed: int = 0,
            warm_start_drafts: int | None = None,
        ) -> None:
            require_leakage_safe_config(cfg)
            super().__init__(cfg, task_info)
            self.operator_policy = operator_policy or RoundRobinOperatorPolicy()
            self.node_selector = node_selector or BestValidationSelector()
            self.policy_rng = random.Random(policy_seed)
            self.warm_start_drafts = (
                cfg.num_drafts if warm_start_drafts is None else warm_start_drafts
            )
            self.last_policy_transition: PolicyTransition | None = None

        def _eligible_node_map(self) -> dict[OperatorName, list[Any]]:
            debuggable = [
                node
                for node in self.journal.buggy_nodes
                if node.is_leaf and node.debug_depth <= self.cfg.max_debug_depth
            ]
            good = list(self.journal.good_nodes)
            return {
                "draft": [],
                "debug": debuggable,
                "improve": good,
            }

        def _make_context(
            self, eligible: Mapping[OperatorName, Sequence[Any]]
        ) -> SearchContext:
            evaluation_limit = max(1, int(self.cfg.step_limit) - 1)
            evaluations_used = max(0, int(self.state.current_step) - 1)
            remaining_evaluations = max(0, evaluation_limit - evaluations_used)
            step_fraction = evaluations_used / evaluation_limit
            time_fraction = self.state.running_time / max(
                1.0, float(self.cfg.time_limit_secs)
            )
            eligible_names = tuple(
                name for name, nodes in eligible.items() if name == "draft" or nodes
            )
            return SearchContext(
                step=self.state.current_step,
                step_limit=self.cfg.step_limit,
                evaluations_used=evaluations_used,
                evaluation_limit=evaluation_limit,
                remaining_evaluations=remaining_evaluations,
                elapsed_secs=float(self.state.running_time),
                time_limit_secs=float(self.cfg.time_limit_secs),
                budget_fraction=min(1.0, max(step_fraction, time_fraction)),
                draft_count=len(self.journal.draft_nodes),
                good_count=len(self.journal.good_nodes),
                buggy_count=len(self.journal.buggy_nodes),
                eligible_operators=eligible_names,
                nodes=tuple(online_view(node) for node in self.journal.nodes),
            )

        def _choose_decision(self) -> SearchDecision:
            eligible = self._eligible_node_map()
            context = self._make_context(eligible)

            if len(self.journal.draft_nodes) < self.warm_start_drafts:
                operator = "draft"
                reason = "fixed_warm_start"
            else:
                operator = self.operator_policy.choose(context, self.policy_rng)
                reason = "operator_policy"

            if operator not in context.eligible_operators:
                raise RuntimeError(f"Policy selected infeasible operator {operator!r}")

            parent_ids: tuple[str, ...] = ()
            if operator in {"debug", "improve"}:
                candidates = [online_view(node) for node in eligible[operator]]
                (selected,) = self.node_selector.select(
                    context, candidates, 1, self.policy_rng
                )
                parent_ids = (selected.node_id,)

            return SearchDecision(
                operator=operator,
                parent_ids=parent_ids,
                policy_name=self.operator_policy.name,
                diagnostics={
                    "reason": reason,
                    "node_selector": self.node_selector.name,
                    "eligible_operators": list(context.eligible_operators),
                    "budget_fraction": context.budget_fraction,
                },
            )

        def _resolve_parent(self, node_id: str) -> Any:
            for node in self.journal.nodes:
                if node.id == node_id:
                    return node
            raise KeyError(f"Policy selected unknown node {node_id}")

        @staticmethod
        def _metric_value(node: Any | None) -> float | None:
            if node is None or node.metric is None or node.metric.value is None:
                return None
            return float(node.metric.value)

        def save_checkpoint(self) -> None:
            """Persist the ordinary solver state plus policy state and RNG."""

            super().save_checkpoint()
            checkpoint_path = Path(self.cfg.checkpoint_path)
            payload = {
                "operator_policy": self.operator_policy.state_dict(),
                "policy_rng_state": self.policy_rng.getstate(),
                "warm_start_drafts": self.warm_start_drafts,
            }
            (checkpoint_path / "policy.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        def load_checkpoint(self) -> None:
            """Restore policy state when a matching sidecar is available."""

            super().load_checkpoint()
            policy_path = Path(self.cfg.checkpoint_path) / "policy.json"
            if not policy_path.exists():
                return
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
            self.operator_policy.load_state_dict(payload["operator_policy"])
            self.policy_rng.setstate(_nested_tuple(payload["policy_rng_state"]))
            self.warm_start_drafts = int(payload["warm_start_drafts"])

        def step(self, task: Any, state: Any):
            """Dispatch a factorized policy decision through stock Dojo mechanics."""

            if not self.journal.nodes or self.data_preview is None:
                self.update_data_preview(state)

            incumbent_before_node = self.journal.get_best_node()
            incumbent_before = self._metric_value(incumbent_before_node)
            decision = self._choose_decision()

            if decision.operator == "draft":
                result_node = self._draft()
            elif decision.operator == "debug":
                result_node = self._debug(self._resolve_parent(decision.parent_ids[0]))
            elif decision.operator == "improve":
                result_node = self._improve(
                    self._resolve_parent(decision.parent_ids[0])
                )
            else:
                raise NotImplementedError(
                    f"Operator {decision.operator!r} is not wired in the minimal scaffold"
                )

            state, eval_result = task.step_task(state, extract_code(result_node.code))
            self.parse_eval_result(node=result_node, eval_result=eval_result)
            self.journal.append(result_node)

            incumbent_after_node = self.journal.get_best_node()
            incumbent_after = self._metric_value(incumbent_after_node)
            if incumbent_before is None or incumbent_after is None:
                incumbent_delta = None
            elif self.lower_is_better:
                incumbent_delta = incumbent_before - incumbent_after
            else:
                incumbent_delta = incumbent_after - incumbent_before

            transition = PolicyTransition(
                step=self.state.current_step,
                decision=decision,
                child_id=result_node.id,
                child_validation_metric=self._metric_value(result_node),
                child_is_buggy=bool(result_node.is_buggy),
                incumbent_before=incumbent_before,
                incumbent_after=incumbent_after,
                incumbent_delta=incumbent_delta,
                execution_time=result_node.exec_time,
            )
            self.operator_policy.observe(transition)
            self.last_policy_transition = transition

            best_step = 0 if incumbent_after_node is None else incumbent_after_node.step
            self.logger.log(asdict(transition), "POLICY", step=self.state.current_step)
            self.logger.log(
                self.journal.get_node_data(self.state.current_step)
                | {"current_best_node": best_step},
                "JOURNAL",
                step=self.state.current_step,
            )
            self.logger.log(
                self.state.state_dict(), "STATE", step=self.state.current_step
            )
            return state, eval_result

    PolicyDrivenGreedy.__name__ = "PolicyDrivenGreedy"
    PolicyDrivenGreedy.__qualname__ = "PolicyDrivenGreedy"
    return PolicyDrivenGreedy


def _demo_context(eligible: Sequence[str], step: int = 5) -> SearchContext:
    nodes = (
        OnlineNodeView("a", 1, 0.40, True, False, True, 1, 0, ("draft",), 0.1),
        OnlineNodeView("b", 2, 0.70, True, False, True, 2, 0, ("improve",), 0.2),
    )
    evaluations_used = max(0, step - 1)
    evaluation_limit = 9
    return SearchContext(
        step=step,
        step_limit=10,
        evaluations_used=evaluations_used,
        evaluation_limit=evaluation_limit,
        remaining_evaluations=max(0, evaluation_limit - evaluations_used),
        elapsed_secs=float(step),
        time_limit_secs=10.0,
        budget_fraction=evaluations_used / evaluation_limit,
        draft_count=2,
        good_count=2,
        buggy_count=0,
        eligible_operators=tuple(eligible),
        nodes=nodes,
    )


def self_test() -> None:
    try:
        require_leakage_safe_config(SimpleNamespace(use_test_score=True))
    except ValueError:
        pass
    else:
        raise AssertionError("private-test-score mode was not rejected")

    checkpoint_rng = random.Random(19)
    restored_rng = random.Random()
    restored_rng.setstate(
        _nested_tuple(json.loads(json.dumps(checkpoint_rng.getstate())))
    )
    assert [checkpoint_rng.random() for _ in range(4)] == [
        restored_rng.random() for _ in range(4)
    ]

    public_node = dict(
        id="same",
        step=1,
        is_buggy=False,
        is_leaf=True,
        parents=[],
        debug_depth=0,
        operators_used=["draft"],
        exec_time=0.1,
        code="__result__ = 1",
        exit_code=0,
    )
    low_private = SimpleNamespace(
        **public_node,
        metric=SimpleNamespace(value=0.5, maximize=True, info={"score": 0.01}),
    )
    high_private = SimpleNamespace(
        **public_node,
        metric=SimpleNamespace(value=0.5, maximize=True, info={"score": 0.99}),
    )
    assert online_view(low_private) == online_view(high_private)

    rng = random.Random(7)
    selector = BestValidationSelector()
    context = _demo_context(("draft", "improve"))
    selected = selector.select(context, context.nodes, 1, rng)
    assert selected[0].node_id == "b"

    minimizing = (
        OnlineNodeView("x", 1, 0.3, False, False, True, 1, 0, (), None),
        OnlineNodeView("y", 2, 0.1, False, False, True, 1, 0, (), None),
    )
    assert selector.select(context, minimizing, 1, rng)[0].node_id == "y"

    buggy = (
        OnlineNodeView("d1", 3, None, True, True, True, 1, 1, ("draft",), None),
        OnlineNodeView("d2", 4, None, True, True, True, 2, 1, ("draft",), None),
    )
    stock_selector = StockGreedyNodeSelector()
    assert stock_selector.select(context, buggy, 1, random.Random(3))[0] in buggy
    improve_context = _demo_context(("improve",))
    assert (
        stock_selector.select(improve_context, improve_context.nodes, 1, rng)[0].node_id
        == "b"
    )

    policy = RoundRobinOperatorPolicy()
    sequence = [
        policy.choose(_demo_context(("draft", "improve"), step), rng)
        for step in range(3)
    ]
    assert sequence == ["draft", "improve", "draft"]

    saved = policy.state_dict()
    restored = RoundRobinOperatorPolicy(("debug",))
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    print("adaptive_policy_scaffold: self-test passed")


def demo() -> None:
    rng = random.Random(11)
    policies: list[OperatorPolicy] = [
        FixedGreedyOperatorPolicy(debug_probability=1.0),
        RandomOperatorPolicy(),
        RoundRobinOperatorPolicy(),
    ]
    contexts = [
        _demo_context(("draft",), 1),
        _demo_context(("draft", "improve"), 4),
        _demo_context(("draft", "improve", "debug"), 8),
    ]
    rows = []
    for policy in policies:
        for context in contexts:
            rows.append(
                {
                    "policy": policy.name,
                    "step": context.step,
                    "eligible": list(context.eligible_operators),
                    "chosen": policy.choose(context, rng),
                }
            )
    print(json.dumps(rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="run dependency-free policy checks"
    )
    parser.add_argument(
        "--check-dojo",
        action="store_true",
        help="verify that the Dojo adapter can be constructed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.check_dojo:
        cls = make_policy_driven_greedy_class()
        print(f"Dojo adapter available: {cls.__name__}")
    else:
        demo()


if __name__ == "__main__":
    main()
