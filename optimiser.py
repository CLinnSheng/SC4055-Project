"""
Classical optimisation loop for one-layer QAOA and XQAOA.

References
----------
[1] V. Vijendran et al., "An Expressive Ansatz for Low-Depth Quantum
    Approximate Optimisation", arXiv:2302.04479v2 (2024).
"""

import time
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from circuits import (
    Angles,
    Edges,
    cut_values,
    expected_score,
    qaoa_circuit,
    xqaoa_circuit,
)

Method = Literal["qaoa", "xqaoa"]
Settings = Angles


def num_settings(method: Method, n: int, edges: Edges) -> int:
    """
    QAOA: 2 Settings
    XQAOA (XY Mixer): m + 2n settings, [1] Table I
    """
    if method == "qaoa":
        return 2
    if method == "xqaoa":
        return len(edges) + 2 * n

    raise ValueError(f"Unknown Method: {method}")


def build_circuit(
    method: Method, settings: Settings, n: int, edges: Edges
) -> QuantumCircuit:
    expected_count = num_settings(method, n, edges)
    if len(settings) != expected_count:
        raise ValueError(
            f"{method} expects {expected_count} settings, got {len(settings)}"
        )

    if method == "qaoa":
        gamma, beta = settings
        return qaoa_circuit(gamma, beta, n, edges)

    m = len(edges)
    gammas = settings[:m]
    betas = settings[m : m + n]
    alphas = settings[m + n : m + 2 * n]

    return xqaoa_circuit(gammas, betas, alphas, n, edges)


class Objective:
    """The function the optimiser calls. Returns the NEGATIVE average score,
    because SciPy minimises and we want the score as high as possible.
    Counts every evaluation and records the score each time."""

    def __init__(self, method: Method, n: int, edges: Edges) -> None:
        self.method: Method = method
        self.n = n
        self.edges = edges
        self.values: NDArray[np.float64] = cut_values(n, edges)
        self.n_evaluations: int = 0
        self.history: list[float] = []

    def score(self, settings: Settings) -> float:
        """Average score for these settings"""
        circuit = build_circuit(self.method, settings, self.n, self.edges)
        return expected_score(circuit, self.values)

    def __call__(self, settings: Settings) -> float:
        """Score it, count it, record it, return the negative"""
        s = self.score(settings)
        self.n_evaluations += 1
        self.history.append(s)

        # Negative because SciPy minimises, and we want to maximise the score
        return -s


def random_start(
    method: Method, n: int, edges: Edges, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Random starting settings: gammas in [0, 2pi), betas and alphas in [0, pi).
    These ranges cover every distinct circuit, [1] footnote 6.
    Same order as build_circuit: [gammas, betas, alphas]."""
    if method == "qaoa":
        return np.array([rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)])

    m = len(edges)
    return np.concatenate(
        [
            rng.uniform(0, 2 * np.pi, m),  # one gamma per edge
            rng.uniform(0, np.pi, n),  # one beta per transmitter
            rng.uniform(0, np.pi, n),  # one alpha per transmitter
        ]
    )


class TrainResult(TypedDict):
    """Everything task 4 needs to save for one training run."""

    method: Method
    seed: int
    initial_settings: list[float]
    final_settings: list[float]
    expected_score: float
    n_evaluations: int
    runtime_s: float
    history: list[float]


def train(
    method: Method, n: int, edges: Edges, max_evaluations: int, seed: int
) -> TrainResult:
    """Train one method from one random start. Using COBYLA method and
    same evaluation limit for both methods, so the comparison is fair."""
    rng = np.random.default_rng(seed)
    start = random_start(method, n, edges, rng)
    objective = Objective(method, n, edges)

    t0 = time.perf_counter()
    result = minimize(
        objective,
        start,
        method="COBYLA",
        options={"maxiter": max_evaluations},
    )
    runtime = time.perf_counter() - t0

    return {
        "method": method,
        "seed": seed,
        "initial_settings": start.tolist(),
        "final_settings": result.x.tolist(),
        "expected_score": objective.score(result.x),
        "n_evaluations": objective.n_evaluations,
        "runtime_s": runtime,
        "history": objective.history,
    }
