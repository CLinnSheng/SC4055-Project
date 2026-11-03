"Exact MaxCut references for the six graphs in the quantum validation tests."
 
from itertools import product
from pathlib import Path
import csv
import json

NETWORKS = {
    "single_edge": (2, [(0, 1)]),
    "line": (3, [(0, 1), (1, 2)]),
    "triangle": (3, [(0, 1), (1, 2), (2, 0)]),
    "star": (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
    "two_triangles": (5, [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]),
    "prism": (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                  (0, 3), (1, 4), (2, 5)]),
}


def cut_score(assignment, edges):
    """Count edges connecting different bits; assignment[i] belongs to node i."""
    return sum(assignment[u] != assignment[v] for u, v in edges)


def solve(n, edges):
    """Check all 2**n assignments, retaining every optimum and every score."""
    if n < 1:
        raise ValueError("n must be positive")
    undirected = [tuple(sorted(edge)) for edge in edges]
    if (any(u == v or not (0 <= u < n and 0 <= v < n) for u, v in edges)
            or len(set(undirected)) != len(edges)):
        raise ValueError("Expected unique, non-self edges with valid node labels")
    all_scores = []
    for assignment in product((0, 1), repeat=n):
        all_scores.append({
            "assignment": "".join(map(str, assignment)),
            "score": cut_score(assignment, edges),
        })
    optimum = max(row["score"] for row in all_scores)
    optimal = [row["assignment"] for row in all_scores if row["score"] == optimum]
    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "edges": edges,
        "assignments_checked": len(all_scores),
        "optimal_score": optimum,
        "optimal_assignments": optimal,
        "n_optimal_assignments": len(optimal),
        "random_expected_score": sum(row["score"] for row in all_scores) / len(all_scores),
        "all_scores": all_scores,
    }


def main():
    """Save reproducible JSON and CSV outputs beside this module."""
    output = Path(__file__).resolve().parent / "reference_results"
    output.mkdir(exist_ok=True)
    results = {}
    summary = []
    for name, (n, edges) in NETWORKS.items():
        result = solve(n, edges)
        results[name] = result
        summary.append({"network": name, **{key: result[key] for key in (
            "n_nodes", "n_edges", "assignments_checked", "optimal_score",
            "n_optimal_assignments", "random_expected_score")}})
        with (output / f"{name}_assignments.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["assignment", "score"])
            writer.writeheader()
            writer.writerows(result["all_scores"])
        print(f"{name}: optimum={result['optimal_score']}, "
              f"optimal strings={len(result['optimal_assignments'])}, "
              f"assignments checked={result['assignments_checked']}")
    (output / "exact_references.json").write_text(json.dumps(results, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
