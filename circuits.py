"""
One-layer QAOA and XQAOA circuits for MaxCut.

References
----------
[1] V. Vijendran, A. Das, D. E. Koh, S. M. Assad and P. K. Lam,
    "An Expressive Ansatz for Low-Depth Quantum Approximate Optimisation",
    arXiv:2302.04479v2 (2024). https://arxiv.org/abs/2302.04479
[2] Qiskit gate definitions (RZZGate, RXGate, RYGate):
    https://docs.quantum.ibm.com/api/qiskit/circuit_library

Angle convention: [1] writes gates as exp(-i*angle*operator), while Qiskit
uses half-angles, e.g. RX(theta) = exp(-i*theta/2*X) [2]. So the paper's
exp(-i*beta*X) is rx(2*beta), and the paper's cost gate
exp(-i*gamma*(1 - Z_u Z_v)/2) is rzz(-gamma), up to a global phase.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

Edges = Sequence[tuple[int, int]]
Angles = Sequence[float] | NDArray[np.float64]


def qaoa_circuit(gamma: float, beta: float, n: int, edges: Edges) -> QuantumCircuit:
    """One-layer standard QAOA, [1] Sec. II B, Eqs. (4)-(8), Fig. 1."""
    qc = QuantumCircuit(n)

    # Part 1
    qc.h(range(n))

    # Part 2 (Cost)
    for u, v in edges:
        qc.rzz(-gamma, u, v)

    # Part 3 (Mixer)
    qc.rx(2 * beta, range(n))

    return qc


def xqaoa_circuit(
    gammas: Angles, betas: Angles, alphas: Angles, n: int, edges: Edges
) -> QuantumCircuit:
    """One-layer XQAOA with the full XY mixer, [1] Sec. III,
    Eqs. (15), (19)-(21), Fig. 1c."""
    if len(gammas) != len(edges) or len(betas) != n or len(alphas) != n:
        raise ValueError(
            f"expected {len(edges)} gammas and {n} betas/alphas, "
            f"got {len(gammas)}, {len(betas)}, {len(alphas)}"
        )

    qc = QuantumCircuit(n)

    # Part 1
    qc.h(range(n))

    # Part 2 (Cost): Each edge has its own gamma
    for (u, v), g in zip(edges, gammas):
        qc.rzz(-g, u, v)

    # Part 3 (Mixer): Each qubit has its own X rotation then its own Y rotation
    for j in range(n):
        qc.rx(2 * betas[j], j)
        qc.ry(2 * alphas[j], j)

    return qc


def cut_values(n: int, edges: Edges) -> NDArray[np.float64]:
    """Scores of every assignment: how many interfering pairs are on different
    channels, [1] Eqs. (2) and (12). Bit i of the index is transmitter i."""
    values = np.zeros(2**n)
    for index in range(2**n):
        for u, v in edges:
            bit_u = (index >> u) & 1
            bit_v = (index >> v) & 1

            # Different Channel
            if bit_u != bit_v:
                values[index] += 1

    return values


def expected_score(circuit: QuantumCircuit, values: NDArray[np.float64]) -> float:
    """Average score of the circuit, [1] Eq. (9)."""
    probs = Statevector(circuit).probabilities()
    return float(probs @ values)


def to_assignment(index: int, n: int) -> str:
    """Turn an assignment number into the project's standard string:
    character i is the channel of node i (node 0 first, read left to right).
    This matches classical_reference.py and reference_results/."""
    return "".join(str((index >> i) & 1) for i in range(n))


def from_qiskit_bitstring(bitstring: str) -> str:
    """Qiskit writes node 0 as the LAST character. Reverse it so that
    character i is node i, the project's standard order."""
    return bitstring[::-1]
