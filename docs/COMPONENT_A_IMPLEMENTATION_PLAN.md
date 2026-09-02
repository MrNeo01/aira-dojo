# FAS Component A: Minimum Integration Plan

## Goal

Add the Component-A operator policy from **Factorized Adaptive Search (FAS)**
to AIRA-dojo as a research proof-of-concept. A run must use the existing AIRA
Draft, Debug, Improve, Analysis, execution, and evaluation code; only the
operator-allocation decision changes.

## Scope frozen for the proof-of-concept

- Arms: `draft`, `debug`, and `improve`.
- Warm start: the existing `num_drafts` Draft actions, charged to the run but
  excluded from bandit evidence.
- Parent choice: validation-best good node for Improve and seeded-uniform
  eligible buggy leaf for Debug.
- Budget and reward cost: one completed candidate evaluation is one unit.
- Phases: early, middle, and late, based on the fraction of the root-excluded
  candidate-evaluation budget already consumed.
- Online signal: direction-corrected improvement in the global validation
  incumbent. Private test fields are never read, and `use_test_score=True` is
  rejected.

Crossover is excluded because the current Greedy solver does not expose the
same Crossover operator as every controlled baseline. Analysis remains
mandatory overhead rather than an arm. Backtracking belongs to parent/node
selection and is not a code-producing arm.

## Minimal implementation steps

1. Add a self-contained `dojo.components.component_a` package.
   - Define the three phases and phase mapping.
   - Store one common sliding action/reward window per phase.
   - Track forced-once exploration per `(phase, arm)` independently of window
     eviction.
   - Implement the manuscript's mean fitness-rate, average-rank decay, and
     smoothed UCB equations with seeded-uniform tie breaking.
   - Add JSON-safe policy state save/load and direction-aware reward helpers.
2. Add `ComponentASolver`, a narrow subclass of `Greedy`.
   - Reuse setup, operators, evaluator parsing, Journal, and run termination.
   - Preserve the Greedy `step()` evaluation and Journal sequence while
     replacing its decision/dispatch block and adding policy observation/logging.
   - Build an explicit feasibility mask at every decision.
   - Observe the bandit only after the generated child has been evaluated.
   - Emit a separate `POLICY` record containing the mask, phase, scores, reward,
     and selected parent/operator.
3. Add a typed solver config and register it in `SOLVER_MAP`.
4. Add generic and MLE-bench Hydra YAMLs that reuse the current AIRA Greedy
   operator prompts and clients unchanged.
5. Add tests and a dependency-free demonstration.
   - Formula, ranking, phases, masking, forced exploration, window eviction,
     seeded ties, reward direction, state round-trip, and leakage guard.
   - A deterministic changing-reward task must show that allocation moves to
     the best arm in each phase and beats a phase-agnostic round-robin control.
   - Compile all changed Python files.
   - Run the zero-API Dojo toy adapter in an AIRA-dojo Python environment to
     exercise the real solver/Journal path.

## Files

New implementation:

```text
src/dojo/components/__init__.py
src/dojo/components/component_a/__init__.py
src/dojo/components/component_a/bandit.py
src/dojo/components/component_a/solver.py
src/dojo/config_dataclasses/solver/component_a.py
src/dojo/configs/solver/component_a.yaml
src/dojo/configs/solver/mlebench/component_a.yaml
src/dojo/configs/_exp/fas_a_run_example.yaml
src/dojo/core/solvers/utils/action_mask.py
examples/fas_component_a_poc.py
tests/component_a/test_bandit.py
docs/COMPONENT_A_IMPLEMENTATION_PLAN.md
docs/FACTORIZED_ADAPTIVE_SEARCH_COMPONENT_A.md
```

Existing code modified:

```text
src/dojo/config_dataclasses/solver/__init__.py  # solver registration only
src/dojo/main_runner_job_array.py                # keep policy seed paired in sweeps
src/dojo/solvers/greedy/greedy.py                # shared corrected Debug mask/export label
examples/toy_search_demo.py                      # zero-API real-Dojo smoke path
docs/ADAPTIVE_POLICY_FOUNDATION.md                # implementation status/link
README.md                                         # documentation links
```

No task, operator, Journal, evaluator, MCTS, or Evolutionary implementation
needs to change. Stock Greedy's generation/evaluation behavior remains the
same; it now calls the shared corrected Debug-feasibility helper and uses a
class-level export label so controlled Greedy and FAS-A runs share the same
action semantics without being mislabeled.

## Success criteria

- Component A starts from empty online statistics and needs no prior logs or
  offline training.
- Every selected arm is feasible at decision time.
- Each newly feasible arm is forced exactly once per phase, even after its
  event leaves the sliding window.
- Computed credits and UCB values match the manuscript equations.
- Maximize and minimize tasks produce the correct non-negative progress reward.
- The Hydra config resolves to `ComponentASolverConfig`, and the registry maps
  it to `ComponentASolver`.
- Configuration rejects a warm start that would consume every candidate
  evaluation, guaranteeing at least one adaptive decision.
- The deterministic proof-of-concept passes and visibly changes operator
  allocation as rewards change by phase.

## Deliberately deferred

- Token-, latency-, or monetary-cost conversion and a matching non-step budget.
- Replayed identical warm-start artifacts across independent processes.
- Crossover support across all controlled policies.
- Exact whole-run resume beyond the upstream checkpoint guarantees.
- Distributed execution hardening, dashboards, and production telemetry.
- MLE-bench performance claims; those require matched prompts/models/budgets,
  multiple tasks and seeds, confidence intervals, and a pre-registered proxy
  study after this software proof succeeds.
