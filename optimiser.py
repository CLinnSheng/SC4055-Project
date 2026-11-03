"""
Classical optimisation loop for one-layer QAOA and XQAOA.

References
----------
[1] V. Vijendran et al., "An Expressive Ansatz for Low-Depth Quantum
    Approximate Optimisation", arXiv:2302.04479v2 (2024).
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from circuits import cut_values, expected_score, qaoa_circuit, xqaoa_circuit

Method = Literal["qaoa", "xqaoa"]
Edges = list[tuple[int, int]]
Settings = Sequence[float] | NDArray[np.float64]


def num_settings(method: Method, n: int, edges: Edges) -> int:
    """
    QAOA: 2 Settings
    XQAOA (XY Mixer): m + 2n settings, refer from [1]
    """
    if method == "qaoa":
        return 2
    if method == "xqaoa":
        return len(edges) + 2 * n

    raise ValueError(f"Unknown Method: {method}")


def build_circuit(
    method: Method, settings: Settings, n: int, edges: Edges
) -> QuantumCircuit:
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
