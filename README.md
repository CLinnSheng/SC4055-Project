# Comparing QAOA and XQAOA for MaxCut on Small Networks

Does full XY XQAOA achieve higher expected MaxCut scores than standard QAOA on
small graphs when both use one layer and the same maximum number of training
evaluations?

Based on: V. Vijendran et al., "An Expressive Ansatz for Low-Depth Quantum
Approximate Optimisation", arXiv:2302.04479v2 (2024).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python version: 3.11 (see `.python-version`).
Exact package versions are pinned in `uv.lock`.

```
uv sync
```

## Run the tests

```
uv run pytest -v
```

## Reproduce the results

Classical reference answers (exact best score for each graph):

```
uv run python classical_reference.py
```

Quantum experiments (about 30 seconds):

```
uv run python run_experiment.py
```

Scores and evaluation counts are identical on every rerun because the seeds are
fixed. Runtimes depend on the machine.

## Experiment settings

| Setting | Value |
|---|---|
| Methods | One-layer standard QAOA, one-layer full XY XQAOA |
| Graphs | single_edge, line, triangle, star, two_triangles, prism (defined in `classical_reference.py`) |
| Starting settings | 5 per method and graph, seeds 0, 1, 2, 3, 4 |
| Start ranges | gamma in [0, 2pi), beta and alpha in [0, pi) |
| Optimiser | SciPy `minimize`, method `COBYLA`, default tolerances |
| Maximum evaluations | 1000 per training run, same for both methods |
| Simulator | Qiskit `Statevector` (exact expected score, no sampling) |
| Total runs | 6 graphs x 2 methods x 5 seeds = 60 |

## Files

| File | Contents |
|---|---|
| `circuits.py` | QAOA and XQAOA circuits, assignment scores, expected score |
| `optimiser.py` | Training loop (COBYLA) |
| `analytic.py` | The paper's exact formulas, Eqs. (14) and (22), used only in tests |
| `classical_reference.py` | Graph definitions and exhaustive exact solver |
| `run_experiment.py` | Runs all 60 trainings |
| `reference_results/` | Exact best scores and all optimal assignments |
| `results/runs.json` | One record per training run: scores, evaluations, runtime, starting and final settings, history |
| `tests/` | Validation tests |
