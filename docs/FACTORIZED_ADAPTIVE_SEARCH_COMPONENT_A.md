# Factorized Adaptive Search (FAS): Component A

## What is implemented

FAS Component A (FAS-A) is an optional online operator-selection layer for
AIRA-dojo. It decides whether the next evaluated candidate should be produced
by **Draft**, **Debug**, or **Improve**. It starts with empty statistics on every
run and needs neither model training nor previous logs.

The implementation is a research proof-of-concept. It keeps the existing AIRA
operator prompts, LLM clients, memory, Analysis call, execution environment,
validation parser, Journal, and stopping loop unchanged. This isolates the
effect of operator allocation.

## Decision rule

1. Generate the common `num_drafts` warm start. These evaluations consume the
   budget but are not treated as bandit-selected evidence.
2. Compute the phase from the consumed candidate-evaluation budget:
   `early < 1/3`, `middle < 2/3`, otherwise `late`.
3. Build an action mask:
   - Draft is always feasible.
   - Debug needs a buggy leaf below `max_debug_depth`.
   - Improve needs at least one valid node.
4. Force every newly feasible arm once per phase. This lifetime phase state is
   separate from the sliding window, so window eviction does not force it again.
5. In each phase, retain one common window of the last `W` adaptive
   `(operator, reward)` events.
6. Compute mean progress rewards for the phase-window arms and rank all observed
   arms with descending average ranks (rank 1 is best). Apply
   `D ** (rank - 1)`, then normalize and score only the currently feasible
   arms. Thus a temporarily masked arm retains its place in the phase ranking
   without being selectable. If every feasible reward is zero, every feasible
   credit is zero.
7. Select the seeded-uniform maximum of:

   ```text
   credit(operator)
   + alpha * sqrt(2 * log(1 + phase_window_events)
                  / (1 + operator_window_events))
   ```

8. Evaluate the child, then update with non-negative, direction-corrected
   global-incumbent progress per cost. The proof-of-concept uses one completed
   candidate evaluation as cost `1.0`, matching AIRA-dojo's current step budget.

For Improve, FAS-A expands the validation-best good node. For Debug, it samples
an eligible buggy leaf with a separate seeded RNG. Keeping policy-tie and
parent-selection RNG streams separate avoids accidental coupling.

## Key implementation choices

| Choice | Reason |
|---|---|
| Subclass `Greedy` | Reuses all existing generation and evaluation mechanics. |
| Three arms only | These operators have matched support in the controlled Greedy path. |
| Common window per phase | Matches the current manuscript's non-stationary bandit definition. |
| Global-incumbent reward | Gives Draft, Debug, and Improve one comparable objective. |
| Unit evaluation cost | It is the only complete scalar budget currently enforced for every action. |
| Reject `use_test_score=True` | Prevents private MLE-bench outcomes from entering online decisions. |
| Warm start excluded from learning | Treats common initialization as charged graph setup, not a policy choice. |

If no valid incumbent exists before an action, its progress reward is zero. We
freeze this proof-of-concept convention instead of inventing a task-dependent
metric floor; a preregistered per-task floor can replace it in the empirical
study if rewarding the first valid solution is required.

## Code layout

```text
src/dojo/components/component_a/bandit.py   # dependency-free FRR-UCB policy
src/dojo/components/component_a/solver.py   # narrow Greedy adapter
src/dojo/core/solvers/utils/action_mask.py   # shared Greedy/FAS-A Debug mask
src/dojo/config_dataclasses/solver/component_a.py
src/dojo/configs/solver/component_a.yaml
src/dojo/configs/solver/mlebench/component_a.yaml
src/dojo/configs/_exp/fas_a_run_example.yaml
examples/fas_component_a_poc.py
examples/toy_search_demo.py                  # lightweight + real Dojo smoke
tests/component_a/test_bandit.py
```

`ComponentASolverConfig` is registered in the existing `SOLVER_MAP`, so the
normal `dojo.main_run` construction path can instantiate it. The config also
requires at least one post-warm-start candidate evaluation so a nominal
Component-A run cannot silently execute only fixed Draft actions.

Small integration changes live in
`src/dojo/config_dataclasses/solver/__init__.py` (registry),
`src/dojo/main_runner_job_array.py` (paired policy seeds),
`src/dojo/solvers/greedy/greedy.py` (shared Debug mask and export label), and
`README.md` (documentation links).

## Run the proof-of-concept

The policy and behavioral test use only the Python standard library:

```bash
python3 -m unittest discover -s tests/component_a -v
python3 examples/fas_component_a_poc.py
python3 examples/toy_search_demo.py --self-test
```

The unit command runs 14 checks. The deterministic example changes the best
operator by phase. With seed 7 and a 90-evaluation budget, FAS-A selected the
correct arm 28/30 times in each phase and obtained synthetic reward `84`,
versus `32` for round-robin. This demonstrates correct adaptation in the
controller; it is not an MLE-bench performance result.

With the normal AIRA-dojo dependencies installed, the following zero-API smoke
drives `ComponentASolver` through the real task protocol, result parser,
Journal, action masks, policy updates, and trace export:

```bash
PYTHONPATH=src python3 examples/toy_search_demo.py \
  --engine dojo --methods component_a \
  --target 101001101010 --budget 18 --seed 23 \
  --output-dir /tmp/fas_component_a_dojo
```

The verified seed-23 smoke completed exactly 18 candidate evaluations and 19
Journal nodes (including the synthetic root), emitted one policy record per
child, and reached toy validation score `0.750`. The example asserts mask
legality, warm-start/learned transition counts, and Journal/operator agreement.
Hydra composition, `ComponentASolverConfig` validation, and the `SOLVER_MAP`
registry mapping were also exercised successfully.

After installing the normal AIRA-dojo environment, configuring the listed LLM
client, MLE-bench data, and the Jupyter/Apptainer interpreter, run the integrated
example through the existing entry point:

```bash
python -m dojo.main_run +_exp=fas_a_run_example logger.use_wandb=False
```

For another experiment, change its solver default from `mlebench/greedy` to
`mlebench/component_a`. Keep prompts, model, task, seed, warm start, and budget
identical when comparing the policies.

## Policy trace

Every evaluated child emits a separate `POLICY` record with:

- selected operator and parent;
- feasibility mask, phase, and consumed-budget fraction;
- warm-start/forced/FRR-UCB selection reason;
- window counts, rates, ranks, credits, and UCB scores;
- validation incumbent before/after, reward, and cost unit;
- child validity/metric and action wall time.

The policy never reads `metric.info`, which can contain private test outcomes.

## Current boundary and next experiment

The implementation proves the layer can allocate feasible AIRA operators and
learn online. It does not yet prove improvement on MLE-bench. The next research
step is a matched-compute proxy study against fixed Greedy, random, and
round-robin controls over multiple tasks and seeds.

Before a paper-level experiment, add a frozen token/time cost conversion that
is also used for stopping, replay identical warm-start artifacts across policy
conditions, keep the zero-API Dojo integration smoke in the regression suite,
and decide whether Crossover can be exposed identically to every controlled
policy. Whole-run checkpoint resume remains limited by the existing AIRA-dojo
resume serializer and should be treated as experimental; paper runs should
start from fresh output directories.
