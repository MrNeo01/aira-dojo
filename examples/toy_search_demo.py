#!/usr/bin/env python3
"""Compare search policies on a tiny non-ML bit-string task.

The objective is to recover a hidden bit string.  Candidate artifacts are
strings, fitness is the fraction of matching positions, and the operators are
draft, improve, debug, and crossover.  This preserves the shape of an AIRA
search run without model training, Kaggle data, GPUs, or API calls.

Two engines are available:

``lightweight`` (default)
    A dependency-free algorithm microscope that mirrors Greedy, rollout-free
    MCTS, and island-style Evolutionary search and writes JSON traces.

``dojo``
    Uses the real Dojo solver loops, Node/Journal, metric parsing, and task
    protocol, while replacing only LLM operators with deterministic toy
    operators.  Run it inside an installed aira-dojo environment.

The toy result is a software/search sanity check, not evidence for Paper 1.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

METHODS = ("greedy", "mcts", "evolutionary")


def make_target(length: int, seed: int) -> str:
    rng = random.Random(seed ^ 0xA17A)
    return "".join(rng.choice("01") for _ in range(length))


def score_bits(bits: str | None, target: str) -> float | None:
    if bits is None or len(bits) != len(target) or set(bits) - {"0", "1"}:
        return None
    return sum(left == right for left, right in zip(bits, target)) / len(target)


def random_bits(length: int, rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(length))


def mutate_bits(bits: str, rng: random.Random, max_flips: int = 3) -> str:
    chars = list(bits)
    flips = rng.randint(1, min(max_flips, len(chars)))
    for index in rng.sample(range(len(chars)), k=flips):
        chars[index] = "1" if chars[index] == "0" else "0"
    return "".join(chars)


def crossover_bits(left: str, right: str, rng: random.Random) -> str:
    child = "".join(rng.choice(pair) for pair in zip(left, right))
    return mutate_bits(child, rng, max_flips=1)


def candidate_code(bits: str | None) -> str:
    return f"__result__ = {bits!r}\n"


def parse_candidate_code(code: str) -> str | None:
    tree = ast.parse(code)
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__result__"
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            return value if isinstance(value, str) else None
    return None


@dataclass
class ToyNode:
    node_id: int
    bits: str | None
    score: float | None
    operator: str
    parent_ids: tuple[int, ...]
    evaluation: int
    is_buggy: bool
    best_so_far: float | None
    visits: int = 0
    value_sum: float = 0.0
    children: list[int] = field(default_factory=list)


class ToyTrace:
    """Small JSON-serializable analogue of Dojo's Journal."""

    def __init__(self, method: str, target: str, budget: int, seed: int) -> None:
        self.method = method
        self.target = target
        self.budget = budget
        self.seed = seed
        self.nodes: list[ToyNode] = [ToyNode(0, None, None, "root", (), 0, True, None)]
        self.evaluations = 0
        self.best_score: float | None = None
        self.best_node_id: int | None = None

    def add(
        self, bits: str | None, operator: str, parent_ids: Sequence[int]
    ) -> ToyNode:
        if self.evaluations >= self.budget:
            raise RuntimeError("Candidate-evaluation budget exhausted")
        self.evaluations += 1
        score = score_bits(bits, self.target)
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
        node = ToyNode(
            node_id=len(self.nodes),
            bits=bits,
            score=score,
            operator=operator,
            parent_ids=tuple(parent_ids),
            evaluation=self.evaluations,
            is_buggy=score is None,
            best_so_far=self.best_score,
        )
        self.nodes.append(node)
        for parent_id in parent_ids:
            self.nodes[parent_id].children.append(node.node_id)
        if score is not None and score == self.best_score:
            self.best_node_id = node.node_id
        return node

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def good_nodes(self) -> list[ToyNode]:
        return [node for node in self.nodes[1:] if not node.is_buggy]

    @property
    def buggy_leaves(self) -> list[ToyNode]:
        return [node for node in self.nodes[1:] if node.is_buggy and not node.children]

    def export(self) -> dict[str, Any]:
        return {
            "engine": "lightweight",
            "method": self.method,
            "task": "hidden_bitstring",
            "target": self.target,
            "candidate_evaluation_budget": self.budget,
            "evaluations_used": self.evaluations,
            "seed": self.seed,
            "best_node_id": self.best_node_id,
            "best_score": self.best_score,
            "nodes": [asdict(node) for node in self.nodes],
        }


class ToyOperators:
    def __init__(self, length: int, seed: int, inject_bug: bool = True) -> None:
        self.length = length
        self.rng = random.Random(seed)
        self.inject_bug = inject_bug
        self.drafts = 0

    def draft(self) -> str | None:
        self.drafts += 1
        if self.inject_bug and self.drafts == 1:
            return None
        return random_bits(self.length, self.rng)

    def improve(self, parent: str) -> str:
        return mutate_bits(parent, self.rng)

    def debug(self, parent: str | None) -> str:
        del parent
        return random_bits(self.length, self.rng)

    def crossover(self, left: str, right: str) -> str:
        return crossover_bits(left, right, self.rng)


def run_lightweight_greedy(target: str, budget: int, seed: int) -> ToyTrace:
    """Mirror Dojo Greedy: warm-start drafts, debug, then best-node improve."""

    trace = ToyTrace("greedy", target, budget, seed)
    operators = ToyOperators(len(target), seed + 101)
    num_drafts = min(4, max(1, budget // 4))

    while trace.remaining:
        draft_nodes = [node for node in trace.nodes[1:] if node.operator == "draft"]
        if len(draft_nodes) < num_drafts:
            trace.add(operators.draft(), "draft", (0,))
            continue
        if trace.buggy_leaves:
            parent = operators.rng.choice(trace.buggy_leaves)
            trace.add(operators.debug(parent.bits), "debug", (parent.node_id,))
            continue
        if trace.good_nodes:
            parent = max(trace.good_nodes, key=lambda node: (node.score, -node.node_id))
            trace.add(
                operators.improve(parent.bits or ""), "improve", (parent.node_id,)
            )
            continue
        trace.add(operators.draft(), "draft", (0,))
    return trace


def _mcts_utility(
    child: ToyNode,
    parent: ToyNode,
    minimum: float,
    maximum: float,
    exploration: float,
) -> float:
    if child.visits == 0:
        return math.inf if not child.is_buggy else -math.inf
    mean = child.value_sum / child.visits
    normalized = 0.5 if minimum == maximum else (mean - minimum) / (maximum - minimum)
    bonus = exploration * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
    return normalized + bonus


def run_lightweight_mcts(target: str, budget: int, seed: int) -> ToyTrace:
    """Approximate AIRA MCTS with UCT selection and no rollout."""

    trace = ToyTrace("mcts", target, budget, seed)
    operators = ToyOperators(len(target), seed + 202)
    branching = 3
    exploration = 0.35
    observed_scores: list[float] = []

    def backup(path: Iterable[ToyNode], value: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += value

    while trace.remaining:
        path = [trace.nodes[0]]
        leaf = path[-1]
        while leaf.children:
            valid_children = [
                trace.nodes[index]
                for index in leaf.children
                if not trace.nodes[index].is_buggy
            ]
            if not valid_children:
                break
            minimum = min(observed_scores, default=0.0)
            maximum = max(observed_scores, default=1.0)
            leaf = max(
                valid_children,
                key=lambda child: _mcts_utility(
                    child, path[-1], minimum, maximum, exploration
                ),
            )
            path.append(leaf)

        children_to_create = min(branching, trace.remaining)
        if children_to_create == 0:
            break
        for _ in range(children_to_create):
            bits = (
                operators.draft()
                if leaf.node_id == 0
                else operators.improve(leaf.bits or "")
            )
            operator = "draft" if leaf.node_id == 0 else "improve"
            child = trace.add(bits, operator, (leaf.node_id,))
            if child.is_buggy:
                if trace.remaining:
                    fixed = trace.add(
                        operators.debug(child.bits), "debug", (child.node_id,)
                    )
                    assert fixed.score is not None
                    observed_scores.append(fixed.score)
                    backup((*path, child, fixed), fixed.score)
                else:
                    backup((*path, child), 0.0)
            else:
                assert child.score is not None
                observed_scores.append(child.score)
                backup((*path, child), child.score)
    return trace


def _weighted_parent(nodes: Sequence[ToyNode], rng: random.Random) -> ToyNode:
    weights = [(node.score or 0.0) + 0.05 for node in nodes]
    return rng.choices(list(nodes), weights=weights, k=1)[0]


def run_lightweight_evolutionary(target: str, budget: int, seed: int) -> ToyTrace:
    """Mirror AIRA Evo with one bounded population and improve/crossover."""

    trace = ToyTrace("evolutionary", target, budget, seed)
    operators = ToyOperators(len(target), seed + 303)
    population_size = min(6, max(2, budget // 4))
    population: list[ToyNode] = []

    while trace.remaining and len(population) < population_size:
        child = trace.add(operators.draft(), "draft", (0,))
        if child.is_buggy and trace.remaining:
            child = trace.add(operators.debug(child.bits), "debug", (child.node_id,))
        if not child.is_buggy:
            population.append(child)

    generation = 1
    while trace.remaining and population:
        use_crossover = (
            generation >= 2 and len(population) >= 2 and operators.rng.random() < 0.5
        )
        if use_crossover:
            left, right = operators.rng.sample(population, k=2)
            bits = operators.crossover(left.bits or "", right.bits or "")
            child = trace.add(bits, "crossover", (left.node_id, right.node_id))
        else:
            parent = _weighted_parent(population, operators.rng)
            child = trace.add(
                operators.improve(parent.bits or ""), "improve", (parent.node_id,)
            )

        if child.is_buggy and trace.remaining:
            child = trace.add(operators.debug(child.bits), "debug", (child.node_id,))
        if not child.is_buggy:
            population.append(child)
            population.sort(key=lambda node: (node.score, -node.node_id), reverse=True)
            del population[population_size:]
        generation += 1
    return trace


LIGHTWEIGHT_RUNNERS: dict[str, Callable[[str, int, int], ToyTrace]] = {
    "greedy": run_lightweight_greedy,
    "mcts": run_lightweight_mcts,
    "evolutionary": run_lightweight_evolutionary,
}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_lightweight(
    methods: Sequence[str],
    target: str,
    budget: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for method in methods:
        trace = LIGHTWEIGHT_RUNNERS[method](target, budget, seed)
        exported = trace.export()
        _write_json(output_dir / f"lightweight_{method}_trace.json", exported)
        results.append(exported)
    return results


def _dojo_import_error(error: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        "The `dojo` engine needs the repository's Python 3.11/3.12 environment. "
        "Create/activate it, run `pip install -e .`, then execute this script with "
        "`PYTHONPATH=src`. The first missing import was "
        f"{error.name!r}. The lightweight engine works without these dependencies."
    )


def run_dojo(
    methods: Sequence[str],
    target: str,
    budget: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run deterministic toy operators through the real Dojo solver classes."""

    os.environ.setdefault("LOGGING_DIR", str(output_dir.resolve()))
    try:
        import numpy as np

        # Import through the registry first.  Importing a concrete solver module
        # directly in this checkout can expose its config/solver circular import.
        from dojo.config_dataclasses.solver import SOLVER_MAP
        from dojo.core.interpreters.base import ExecutionResult
        from dojo.core.solvers.utils.journal import Node
        from dojo.core.tasks.constants import (
            AUX_EVAL_INFO,
            EXECUTION_OUTPUT,
            TASK_DESCRIPTION,
            VALIDATION_FITNESS,
            VALID_SOLUTION,
            VALID_SOLUTION_FEEDBACK,
        )
        from dojo.solvers.mcts.mcts import MCTSNode
        from dojo.utils.logger import config_logger
    except ModuleNotFoundError as error:
        raise _dojo_import_error(error) from error

    Greedy = SOLVER_MAP["GreedySolverConfig"]
    MCTS = SOLVER_MAP["MCTSSolverConfig"]
    Evolutionary = SOLVER_MAP["EvolutionarySolverConfig"]
    config_logger(None)

    class CandidateBudgetExhausted(RuntimeError):
        """Internal signal used to stop a stock loop before an extra evaluation."""

    class AssignmentInterpreter:
        local = True

        def run(self, code: str, **kwargs: Any) -> Any:
            del kwargs
            started = time.monotonic()
            try:
                value = parse_candidate_code(code)
                return ExecutionResult(
                    term_out=[f"__result__={value!r}\n"],
                    exec_time=time.monotonic() - started,
                    exit_code=0,
                    eval_return=value,
                )
            except Exception as error:  # toy execution feedback, intentionally broad
                return ExecutionResult(
                    term_out=[f"{type(error).__name__}: {error}\n"],
                    exec_time=time.monotonic() - started,
                    exit_code=1,
                    eval_return=None,
                )

        def cleanup_session(self) -> None:
            return None

    class BitStringTask:
        def __init__(self) -> None:
            self.candidate_evaluations = 0

        def prepare(self, **kwargs: Any):
            state = {
                "solver_interpreter": kwargs.get(
                    "solver_interpreter", AssignmentInterpreter()
                )
            }
            return state, {
                TASK_DESCRIPTION: "Recover a hidden bit string.",
                "lower_is_better": False,
            }

        def step_task(self, state: dict[str, Any], action: str):
            if self.candidate_evaluations >= budget:
                raise CandidateBudgetExhausted
            self.candidate_evaluations += 1
            result = state["solver_interpreter"].run(action)
            score = (
                score_bits(result.eval_return, target)
                if result.exit_code == 0
                else None
            )
            result.term_out.append(f"TOY_VALIDATION_SCORE={score!r}\n")
            outcome: dict[str, Any] = {
                EXECUTION_OUTPUT: result,
                VALID_SOLUTION: score is not None,
                AUX_EVAL_INFO: {
                    "matches": None if score is None else int(score * len(target))
                },
            }
            if score is None:
                outcome[VALID_SOLUTION_FEEDBACK] = (
                    "__result__ must be a bit string of the configured length"
                )
            else:
                outcome[VALIDATION_FITNESS] = score
            return state, outcome

        def close(self, state: dict[str, Any]) -> None:
            state["solver_interpreter"].cleanup_session()

    class ToyOperatorsMixin:
        toy_node_type: type = Node

        def setup_operators(self) -> None:
            self.toy_rng = random.Random(seed + 1009)
            self.toy_draft_count = 0

        def save_checkpoint(self) -> None:
            return None

        def update_data_preview(self, state: Any) -> None:
            del state
            self.data_preview = "(No dataset: hidden bit-string task)"

        def _new_node(self, bits: str | None, operator: str, parents: Sequence[Any]):
            return self.toy_node_type(
                code=candidate_code(bits),
                plan=f"Toy {operator}",
                parents=list(parents),
                operators_used=[operator],
                operators_metrics=[],
            )

        def _draft(self, parent: Any | None = None):
            self.toy_draft_count += 1
            bits = (
                None
                if self.toy_draft_count == 1
                else random_bits(len(target), self.toy_rng)
            )
            return self._new_node(bits, "draft", [parent or self.root_node])

        def _improve(self, parent_node: Any):
            bits = parse_candidate_code(parent_node.code)
            bits = (
                random_bits(len(target), self.toy_rng)
                if bits is None
                else mutate_bits(bits, self.toy_rng)
            )
            return self._new_node(bits, "improve", [parent_node])

        def _debug(self, parent_node: Any):
            bits = random_bits(len(target), self.toy_rng)
            return self._new_node(bits, "debug", [parent_node])

        def _crossover(self, parent_node1: Any, parent_node2: Any):
            left = parse_candidate_code(parent_node1.code) or random_bits(
                len(target), self.toy_rng
            )
            right = parse_candidate_code(parent_node2.code) or random_bits(
                len(target), self.toy_rng
            )
            return self._new_node(
                crossover_bits(left, right, self.toy_rng),
                "crossover",
                [parent_node1, parent_node2],
            )

        def _analyze(self, node: Any) -> dict[str, Any]:
            node.operators_used.append("analysis")
            node.operators_metrics.append({"toy": True})
            return {
                "metric": None,
                "summary": node.term_out,
                "is_bug": node.exit_code != 0,
            }

    class ToyGreedy(ToyOperatorsMixin, Greedy):
        pass

    class ToyMCTS(ToyOperatorsMixin, MCTS):
        toy_node_type = MCTSNode

        def __call__(self, task: Any, state: Any):
            # The checkout uses <= and can spin when remaining_steps == 0.
            self.create_root_node()
            while self.state.current_step < self.cfg.step_limit:
                before = self.state.current_step
                state = self.step(task, state)
                if self.state.current_step <= before:
                    raise RuntimeError("MCTS made no progress")
            best = self.journal.get_best_node()
            return state, None if best is None else best.code, best

    class ToyEvolutionary(ToyOperatorsMixin, Evolutionary):
        def __call__(self, task: Any, state: Any):
            try:
                return super().__call__(task, state)
            except CandidateBudgetExhausted:
                # Stock Evolutionary checks its limit only after a full
                # generation.  The task-side hard cap stops a partial final
                # generation without evaluating an extra candidate.
                best = self.journal.get_best_node()
                return state, None if best is None else best.code, best

    solver_types = {
        "greedy": ToyGreedy,
        "mcts": ToyMCTS,
        "evolutionary": ToyEvolutionary,
    }

    def common_cfg(method: str, method_dir: Path) -> SimpleNamespace:
        base: dict[str, Any] = {
            # Stock solvers count their synthetic root as step 0/1.  Add one so
            # the CLI budget consistently means evaluated candidate programs.
            "step_limit": budget + 1,
            "time_limit_secs": 60,
            "checkpoint_path": str(method_dir / "checkpoint"),
            "export_search_results": False,
            "exp_name": f"toy_{method}",
            "use_test_score": False,
            "use_complexity": False,
            "max_llm_call_retries": 1,
            "data_preview": False,
            "execution_timeout": 5,
            "operators": {},
            "memory": None,
            "debug_memory": None,
            "available_packages": [],
            "max_debug_depth": 3,
            "max_debug_time": 5,
        }
        if method == "greedy":
            base.update(
                num_drafts=min(4, max(1, budget // 4)),
                debug_prob=1.0,
                improvement_steps=0,
            )
        elif method == "mcts":
            base.update(num_children=3, uct_c=0.35)
        else:
            population = min(5, max(2, budget // 4))
            base.update(
                num_islands=1,
                max_island_size=population,
                crossover_prob=0.5,
                migration_prob=0.0,
                initial_temp=1.0,
                final_temp=0.5,
                num_generations_till_migration=999,
                num_generations_till_crossover=2,
                few_shot={"improve": 1, "crossover": 2},
                num_generations=max(1, math.ceil(budget / population) + 1),
                individuals_per_generation=population,
            )
        return SimpleNamespace(**base)

    results: list[dict[str, Any]] = []
    for method in methods:
        random.seed(seed)
        np.random.seed(seed)
        method_dir = output_dir / f"dojo_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        task = BitStringTask()
        state, task_info = task.prepare(solver_interpreter=AssignmentInterpreter())
        solver = solver_types[method](common_cfg(method, method_dir), task_info)
        state, _, best = solver(task, state)
        task.close(state)
        nodes = solver.journal.node_list()
        if task.candidate_evaluations != budget or len(nodes) != budget + 1:
            raise RuntimeError(
                f"{method} violated the toy candidate budget: "
                f"evaluations={task.candidate_evaluations}, nodes={len(nodes)}, "
                f"budget={budget}"
            )
        running_best: float | None = None
        for raw, node in zip(nodes, solver.journal.nodes):
            extra = node.extra_metrics_to_log()
            if extra:
                raw.update(extra)
            value = None if node.is_buggy or node.metric is None else node.metric.value
            if value is not None:
                running_best = (
                    value if running_best is None else max(running_best, value)
                )
            raw["operator"] = node.operators_used[0] if node.operators_used else "root"
            raw["best_so_far"] = running_best
        exported = {
            "engine": "dojo",
            "method": method,
            "task": "hidden_bitstring",
            "target": target,
            "candidate_evaluation_budget": budget,
            "effective_root_inclusive_step_limit": budget + 1,
            "nodes_generated_including_root": len(nodes),
            "candidate_evaluations": task.candidate_evaluations,
            "seed": seed,
            "best_score": None if best is None else best.metric.value,
            "best_node_id": None if best is None else best.id,
            "nodes": nodes,
        }
        _write_json(output_dir / f"dojo_{method}_trace.json", exported)
        results.append(exported)
    return results


def print_summary(
    results: Sequence[dict[str, Any]], target: str, output_dir: Path
) -> None:
    print(f"target={target}  traces={output_dir.resolve()}")
    print("engine       method          best_score  candidate_evals  nodes")
    print("-----------  --------------  ----------  ---------------  -----")
    for result in results:
        candidate_evals = result.get(
            "evaluations_used", result.get("candidate_evaluations")
        )
        node_count = len(result["nodes"])
        score = result["best_score"]
        score_text = "None" if score is None else f"{score:.3f}"
        print(
            f"{result['engine']:<11}  {result['method']:<14}  {score_text:>10}  "
            f"{candidate_evals:>15}  {node_count:>5}"
        )


def self_test() -> None:
    target = "101001101010"
    assert score_bits(target, target) == 1.0
    assert score_bits(None, target) is None
    assert parse_candidate_code(candidate_code(target)) == target
    changed = mutate_bits(target, random.Random(1))
    assert changed != target and len(changed) == len(target)

    for method, runner in LIGHTWEIGHT_RUNNERS.items():
        trace = runner(target, 18, 23)
        assert trace.evaluations == 18, method
        assert trace.best_score is not None, method
        best_curve = [
            node.best_so_far for node in trace.nodes[1:] if node.best_so_far is not None
        ]
        assert best_curve == sorted(best_curve), method
        assert any(node.operator == "debug" for node in trace.nodes), method
    print("toy_search_demo: self-test passed")


def parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods: {', '.join(unknown)}")
    if not methods:
        raise argparse.ArgumentTypeError("Select at least one method")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("lightweight", "dojo"), default="lightweight"
    )
    parser.add_argument("--methods", type=parse_methods, default=METHODS)
    parser.add_argument("--bits", type=int, default=20, help="hidden target length")
    parser.add_argument("--budget", type=int, default=40, help="candidate evaluations")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target", help="optional explicit target bit string")
    parser.add_argument("--output-dir", type=Path, default=Path("toy_runs"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.bits < 2:
        parser.error("--bits must be at least 2")
    if args.budget < 4:
        parser.error("--budget must be at least 4")
    if args.target is not None and (
        set(args.target) - {"0", "1"} or len(args.target) < 2
    ):
        parser.error("--target must contain at least two binary digits")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    target = args.target or make_target(args.bits, args.seed)
    output_dir = args.output_dir
    if args.engine == "lightweight":
        results = run_lightweight(
            args.methods, target, args.budget, args.seed, output_dir
        )
    else:
        results = run_dojo(args.methods, target, args.budget, args.seed, output_dir)
    print_summary(results, target, output_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
