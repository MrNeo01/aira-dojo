# Foundation for an adaptive policy on AIRA-dojo

## Recommendation

Start from a **policy-driven subclass of Greedy**, not from a new end-to-end
solver and not from MCTS.

Keep the following pieces unchanged:

- AIRA operator prompts and LLM clients;
- task and interpreter;
- code execution and evaluation;
- `parse_eval_result()`;
- `Node`, `Journal`, and search export;
- step and wall-clock termination limits.

Replace only the entangled decision block in `Greedy.step()` with two explicit
hooks:

```text
eligible actions -> operator policy -> node selector -> existing operator
                 -> existing evaluator -> observe transition
```

This is the smallest intervention that can later express the proposal's
factorization into `pi_op` and `pi_sel`. It also gives clean random and
round-robin controls before any learned method is implemented.

[`examples/adaptive_policy_scaffold.py`](../examples/adaptive_policy_scaffold.py)
implements this basis. It intentionally does **not** implement sliding-window
UCB, the hindsight value model, or Paper 1's combined method.

## Why this is the right scope now

The proposal puts the current work in Phase 0:

1. reproduce fixed baselines in one harness;
2. log complete search traces;
3. build cheap proxy tasks;
4. control seeds and quantify variance; and
5. verify that the evaluation pipeline can detect a known policy difference.

Implementing the learned method before those checks would make it impossible to
separate a policy contribution from framework, leakage, budget, or
reproducibility errors.

The scaffold therefore provides comparison policies and extension hooks rather
than the proposed method itself.

## Why overriding `Greedy.search_policy()` is not enough

The current method returns only `Node | None`:

- `None` implicitly means Draft;
- a buggy node implicitly means Debug;
- a good node implicitly means Improve.

That return type cannot represent “choose Improve on node A,” “choose Draft
despite good nodes,” or “choose Crossover on nodes A and B.” It merges node
selection and operator selection.

The scaffold overrides `step()` around the decision/dispatch block but reuses
the stock code from task execution through Journal logging. Its decision object
contains:

```python
SearchDecision(
    operator="draft" | "debug" | "improve",
    parent_ids=(...),
    policy_name="...",
    diagnostics={...},
)
```

This makes the policy boundary explicit without modifying upstream files.

## Interfaces in the scaffold

### `OperatorPolicy`

```python
choose(context, rng) -> operator
observe(transition) -> None
state_dict() -> dict
load_state_dict(state) -> None
```

The included implementations are:

- `FixedGreedyOperatorPolicy`;
- `RandomOperatorPolicy`;
- `RoundRobinOperatorPolicy`.

These are controls and plumbing tests, not the adaptive contribution.

### `NodeSelector`

```python
select(context, candidate_views, count, rng) -> selected_views
```

The included selectors are:

- `BestValidationSelector`, a deterministic best-metric control (and the same
  choice stock Greedy uses for Improve);
- `StockGreedyNodeSelector`, which uses best metric for Improve and seeded
  random choice for feasible Debug parents;
- `RandomNodeSelector`, a cheap control.

The full `SearchContext` supplies budget state and all allowlisted nodes. The
view includes generated code, public error class, and parent IDs, so a future
hindsight selector can derive code embeddings and sibling/history features
without seeing private reports. Component B may still extend this versioned
view, but it can retain the same selector interface.

### `OnlineNodeView`

Policies do not receive raw `Node` objects. They receive an allowlisted view:

- validation metric and direction;
- step, depth, leaf/debug status;
- operator history;
- execution time;
- generated code, parent IDs, and a coarse public error class.

`metric.info` is deliberately omitted because it contains MLE-bench private
test results. In addition, every online adaptive run must set and assert
`cfg.use_test_score == false`: when it is true, `node.metric.value` itself can
be replaced by the private test score. The supplied adapter raises if that
unsafe configuration is passed.

## Run the policy scaffold

The dependency-free policy checks work even before installing Dojo:

```bash
python3 examples/adaptive_policy_scaffold.py --self-test
python3 examples/adaptive_policy_scaffold.py
```

Inside an installed aira-dojo environment, verify that the real subclass can be
imported/generated (this check does not instantiate or step a solver):

```bash
pip install black  # imported by core parsing but absent from requirements.txt
PYTHONPATH=src python3 examples/adaptive_policy_scaffold.py --check-dojo
```

The factory is used as follows in an experiment harness:

```python
from examples.adaptive_policy_scaffold import (
    BestValidationSelector,
    RoundRobinOperatorPolicy,
    make_policy_driven_greedy_class,
)

PolicyDrivenGreedy = make_policy_driven_greedy_class()
solver = PolicyDrivenGreedy(
    cfg,
    task_info,
    operator_policy=RoundRobinOperatorPolicy(),
    node_selector=BestValidationSelector(),
    policy_seed=seed,
    warm_start_drafts=5,
)
```

For a production Hydra experiment, register the resulting solver as described
later in this guide rather than constructing it manually.

## Feasible actions must be explicit

The proposal lists `{draft, debug, improve, crossover, backtrack}`, but these
are not simultaneously legal.

| Action | Feasibility | Parents | Current support |
|---|---|---:|---|
| Draft | Always, or forced during a common warm start | 0 | Greedy/MCTS/Evo |
| Improve | At least one good node | 1 | Greedy/MCTS/Evo |
| Debug | A buggy leaf below the debug-depth limit | 1 | Greedy/MCTS/Evo |
| Crossover | At least two good nodes | 2 | Evo only |
| Backtrack | Select an earlier node/frontier | not a new artifact by itself | No operator |
| Analysis | Mandatory post-execution parsing | current child | All solvers, but not a policy arm |

A future bandit must receive an action mask and choose only among eligible
arms. It must not count an unavailable arm as a failure.

Backtrack is better modeled as a node-selection behavior: select an older node
instead of the current best/frontier node. If it is kept as an operator in the
paper, define exactly what artifact it produces and what cost/reward event it
creates.

The first scaffold supports Draft, Debug, and Improve because those are already
wired identically by Greedy. Add Crossover only after it is available with the
same prompt/operator semantics for every compared method.

## Preserve the exact comparison boundary

For Paper 1, “same operators” must mean literal equality of:

- prompt templates;
- base model and client;
- sampling parameters;
- memory policy;
- analyzer;
- execution environment;
- task/evaluator;
- warm-start candidates or warm-start protocol;
- candidate-evaluation, token, and wall-clock budgets.

Do not compare the stock `AIDE_GREEDY` experiment directly with an AIRA
adaptive experiment and call the difference a search-policy effect; their
prompts differ. Use one chosen operator family for all search-policy rows.

Similarly, Greedy/MCTS without Crossover and Evolutionary with Crossover do not
have identical action spaces. Either add Crossover to every relevant policy or
report that comparison as a compound search-plus-operator baseline rather than
the controlled headline.

## Clean ablation matrix

Once the hooks are stable, one solver can express the required variants:

| Variant | Node selector | Operator policy |
|---|---|---|
| Stock Greedy reproduction | Best for Improve; uniform feasible buggy leaf for Debug | Fixed Greedy rule |
| Deterministic best-parent control | Best validation for both parent-bearing arms | Fixed Greedy rule |
| Random control | Best validation | Uniform over feasible arms |
| Round-robin control | Best validation | Feasible round robin |
| A-only | Best validation | Future phase-aware bandit |
| B-only | Future hindsight-UCB selector | Fixed Greedy rule |
| A+B | Future hindsight-UCB selector | Future phase-aware bandit |

For the first row, instantiate
`FixedGreedyOperatorPolicy(debug_probability=cfg.debug_prob)` together with
`StockGreedyNodeSelector()`. The deterministic best-parent row intentionally
uses `BestValidationSelector()` for both Improve and Debug.

The fixed warm start should be identical across rows. Do not allow one policy to
learn from the warm start while another receives different initial candidates.
For the strongest control, pre-generate and replay the same warm-start nodes per
task/seed when feasible.

## Simplest adaptive method to implement next

After Phase-0 checks pass, implement this **Component-A precursor** first:

1. Keep `BestValidationSelector` unchanged.
2. Force the same 2-5 Draft warm start for all policies.
3. Divide consumed budget into early/mid/late phases.
4. Maintain a sliding reward window per `(phase, operator)`.
5. Force each newly feasible unseen arm once.
6. Select the feasible arm with the largest sliding-window UCB score.
7. Update after the child is evaluated.

This is simpler than the hindsight model because it requires no trace training,
embeddings, model checkpoint, or held-out-task protocol. It also directly tests
the proposal's claim that useful operators change over the run.

This ordinary sliding-window UCB is the simplest useful engineering test, but it
is a deliberate simplification of the proposal. The proposed Component A uses
FRRMAB-style fitness-rate-rank credit with sliding-window UCB. Treat ordinary
SW-UCB as an ablation/precursor; implement and report the FRRMAB-credit variant
before calling a result the proposal-faithful Component-A headline.

## Reward and cost: decide before implementation

The proposal's expression `max(0, delta F) / cost(operator)` leaves several
choices open. Pre-register them before looking at headline results.

### Recommended primary online reward

```text
positive signed improvement in global incumbent / marginal action cost
```

For a higher-is-better task:

```text
max(0, best_after - best_before) / cost
```

For lower-is-better, reverse the subtraction.

Why global incumbent rather than parent-to-child change: the primary evaluation
is an anytime best-so-far curve, so this reward is aligned with the paper's
actual objective and gives Draft/Crossover a meaningful common reference.

Log parent-to-child improvement too, because it is diagnostically useful, but
do not silently switch reward definitions during tuning.

### What action cost must include

At minimum:

- generating-operator LLM tokens and latency;
- mandatory analysis-LLM tokens and latency;
- candidate execution time;
- task evaluation time.

Monetary cost alone is fragile because backend price tables can return zero or
be stale. Record tokens, wall-clock time, and execution time separately. The
headline comparison should enforce the same hard budget and may report several
cost normalizations as ablations.

Draft has no parent baseline; Crossover has two; Debug often turns an invalid
node into a valid one. Global-incumbent improvement avoids ad hoc definitions
for these cases.

## Leakage boundary for the hindsight model

The future value model needs private outcomes to create hindsight labels, but
the online policy must not see them.

Use two explicit stages:

```text
finished run
  -> offline label builder may read private test reports
  -> training dataset with split/task provenance
  -> trained value model
  -> online inference receives allowlisted non-private features only
```

Online features may include:

- `node.metric.value` only when `cfg.use_test_score == false` (the observable
  validation proxy);
- validation history and ranks;
- code embedding;
- public execution/error class;
- depth and sibling statistics;
- operator history;
- remaining budget.

Online features must exclude:

- `node.metric.info["score"]`;
- medal/leaderboard fields;
- any value derived from private answers;
- hindsight labels or descendant test returns from the current evaluation run.

Add a non-leakage test: change every private `metric.info` field while holding
the online state fixed and assert that the selected action is unchanged.

## Trace schema to add during Phase 0

Keep policy records separate from `operators_metrics`. The analysis utilities
assume entries there are LLM-call records.

One `POLICY` record per generated child should contain at least:

```json
{
  "step": 17,
  "child_id": "...",
  "parent_ids": ["..."],
  "operator": "improve",
  "eligible_operators": ["draft", "improve", "debug"],
  "policy_name": "round_robin_operator",
  "node_selector": "best_validation",
  "budget_fraction": 0.42,
  "policy_scores": {},
  "incumbent_before": 0.71,
  "child_validation_metric": 0.74,
  "incumbent_after": 0.74,
  "incumbent_delta": 0.03,
  "generation_tokens": null,
  "analysis_tokens": null,
  "llm_latency_secs": null,
  "execution_time_secs": 31.2,
  "cumulative_cost": null
}
```

The scaffold emits the policy decision and observable transition fields. The
token/cost aggregation still needs production implementation. A `policy.json`
checkpoint sidecar stores policy state, the policy RNG, and warm-start state.
This does not make whole-run resume exact: the upstream solver still recreates
a root and has the Journal/type restoration problems listed in the
implementation guide.

For reproducibility, also save:

- resolved Hydra config and git commit;
- task and dataset versions;
- Python/NumPy/Torch RNG states;
- policy RNG state and sufficient statistics;
- model/client identifiers;
- exact budget counters;
- failure/retry events.

## The non-ML basis

[`examples/toy_search_demo.py`](../examples/toy_search_demo.py) provides two
levels of testing:

1. `--engine lightweight` is dependency-free and makes small approximations of
   the algorithms easy to inspect. It writes one complete JSON graph per method
   and enforces an exact candidate-evaluation budget.
2. `--engine dojo` sends deterministic zero-API operators through the real
   Greedy/MCTS/Evolutionary solver loops, task protocol, result parser, and
   Journal.

Use it as follows:

```bash
python3 examples/toy_search_demo.py --self-test

python3 examples/toy_search_demo.py \
  --engine lightweight \
  --methods greedy,mcts,evolutionary \
  --bits 20 --budget 40 --seed 7 \
  --output-dir toy_runs
```

After installing the repository environment:

```bash
pip install black  # imported by core parsing but absent from requirements.txt
PYTHONPATH=src python3 examples/toy_search_demo.py \
  --engine dojo \
  --methods greedy,mcts,evolutionary \
  --bits 20 --budget 40 --seed 7 \
  --output-dir toy_runs
```

The first draft is intentionally invalid so every method exercises its Debug
path. The objective is deterministic and cheap. The trace should show:

- root and parent-child relationships;
- operator labels;
- invalid candidate and debug child;
- validation metrics;
- monotonically non-decreasing best-so-far score;
- different allocation patterns across Greedy, MCTS, and Evolutionary.

Do not interpret which toy method wins as evidence about MLE-bench. Its purpose
is to expose orchestration and logging errors in seconds. The lightweight MCTS
is intentionally only an approximation (for example, it does not traverse
buggy children the same way as stock Dojo).

The toy Dojo adapter compensates for the synthetic root and applies a task-side
hard cap to Evolutionary, so its `--budget` is an equal number of evaluated
candidates. Stock repository experiments do not have that guard:
Greedy/MCTS count their root and Evolutionary can overshoot a generation. A
shared framework counter is still required before headline comparisons. Seeded
toy operators also do not make entire traces reproducible: node
UUIDs/timestamps vary and MCTS stores children in a set.

## Hardening required before research claims

Resolve these in Phase 0:

1. apply `metadata.seed` to Python, NumPy, Torch, and policy RNGs;
2. change the MCTS outer condition from `<=` to `<` and add a budget test;
3. make task-provided validation fitness authoritative without requiring a
   successful analysis-LLM call;
4. make `main_run` task-generic and call `Task.evaluate_fitness` where needed;
5. replace hard-coded registries with explicit registrations or add all new
   task/solver mappings;
6. add policy/cost trace records;
7. make checkpoint restoration exact or disable resume in experiments;
8. test Evolutionary budget, island IDs, reset direction, and resume state;
9. add a beam baseline if it remains in the paper;
10. freeze a single operator/prompt family across policy comparisons.

## Suggested production layout

Once the external scaffold has passed toy and small API smoke tests, move it
into the package:

```text
src/dojo/solvers/adaptive/
  __init__.py
  adaptive.py
  policies.py
src/dojo/config_dataclasses/solver/adaptive.py
src/dojo/configs/solver/adaptive.yaml
src/dojo/configs/solver/mlebench/adaptive.yaml
tests/
  test_operator_policies.py
  test_policy_leakage.py
  test_toy_solver_integration.py
  test_budget_and_seed_control.py
```

Register `AdaptiveSolverConfig -> AdaptiveSolver` in `SOLVER_MAP`. The
MLE-bench adaptive YAML should load the exact same operator prompt YAMLs and
clients as the corresponding fixed baselines.

## Milestones

### M0: basis supplied here

- architecture understood;
- policy boundaries explicit;
- fixed/random/round-robin controls available;
- dependency-free toy trace generator available;
- real-Dojo zero-API smoke path available.

### M1: framework reliability

- deterministic seeding;
- exact budget accounting;
- generic task runner;
- complete policy/cost trace;
- fixed baseline tests and fresh-run reproducibility.

### M2: Phase-0 empirical gate

- proxy tasks;
- Greedy/MCTS/Evo/beam under one operator family;
- multiple seeds;
- detect a known policy difference above noise.

### M3: Paper-1 Component A

- ordinary phase-aware SW-UCB precursor, then proposal-faithful FRRMAB
  fitness-rate-rank credit with sliding-window UCB;
- random and round-robin controls;
- reward/cost ablations;
- A-only result on proxy tasks.

### M4: Paper-1 Component B

- offline DAG hindsight labels;
- task-disjoint splits;
- uncertainty-aware node selector;
- strict online leakage tests;
- B-only and A+B comparisons.

This ordering keeps the current code useful even if either proposed adaptive
component later fails: the harness, traces, proxy suite, and controlled
baselines remain valid research assets.
