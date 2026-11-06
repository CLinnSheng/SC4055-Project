"""
Train one-layer QAOA and XQAOA on every graph from five starting settings.

Graphs come from classical_reference.py, so the quantum runs and the exact
reference answers always use the same graph definitions.
"""

import json
from pathlib import Path
from typing import TypedDict

from classical_reference import NETWORKS, solve
from optimiser import Method, num_settings, train

METHODS: list[Method] = ["qaoa", "xqaoa"]
SEEDS = [0, 1, 2, 3, 4]
MAX_EVALUATIONS = 1000
RESULTS_DIR = Path("results")


class RunRecord(TypedDict):
    """Everything saved for one training run."""

    graph: str
    method: Method
    seed: int
    n_nodes: int
    n_edges: int
    n_settings: int
    optimal_score: int
    expected_score: float
    approx_ratio: float
    n_evaluations: int
    hit_evaluation_limit: bool
    runtime_s: float
    initial_settings: list[float]
    final_settings: list[float]
    history: list[float]


def run_all() -> list[RunRecord]:
    """Train every method on every graph from every seed."""
    records: list[RunRecord] = []

    for name, (n, edges) in NETWORKS.items():
        best = solve(n, edges)["optimal_score"]

        for method in METHODS:
            for seed in SEEDS:
                result = train(method, n, edges, MAX_EVALUATIONS, seed)

                record: RunRecord = {
                    "graph": name,
                    "method": method,
                    "seed": seed,
                    "n_nodes": n,
                    "n_edges": len(edges),
                    "n_settings": num_settings(method, n, edges),
                    "optimal_score": best,
                    "expected_score": result["expected_score"],
                    "approx_ratio": result["expected_score"] / best,
                    "n_evaluations": result["n_evaluations"],
                    "hit_evaluation_limit": result["n_evaluations"] >= MAX_EVALUATIONS,
                    "runtime_s": result["runtime_s"],
                    "initial_settings": result["initial_settings"],
                    "final_settings": result["final_settings"],
                    "history": result["history"],
                }
                records.append(record)

                print(
                    f"{name:13s} {method:5s} seed {seed}  "
                    f"score {record['expected_score']:.3f}/{best}  "
                    f"({record['approx_ratio']:.1%})  "
                    f"evals {record['n_evaluations']:4d}  "
                    f"time {record['runtime_s']:.2f}s"
                )

    return records


def save_results(records: list[RunRecord], out_dir: Path) -> None:
    """Write every run, including history, to runs.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "runs.json", "w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    records = run_all()
    save_results(records, RESULTS_DIR)
    print(f"\nSaved {len(records)} runs to {RESULTS_DIR / 'runs.json'}")


if __name__ == "__main__":
    main()
