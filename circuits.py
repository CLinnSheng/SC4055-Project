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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def qaoa_circuit(gamma, beta, n, edges):
    qc = QuantumCircuit(n)

    # Part 1
    qc.h(range(n))

    # Part 2 (Cost)
    for u, v in edges:
        qc.rzz(-gamma, u, v)

    # Part 3 (Mixer)
    qc.rx(2 * beta, range(n))

    return qc


def xqaoa_circuit(gamma, betas, alphas, n, edges):
    qc = QuantumCircuit(n)

    # Part 1
    qc.h(range(n))

    # Part 2 (Cost): Each edge has its own gamma
    for (u, v), g in zip(edges, gamma):
        qc.rzz(-g, u, v)

    # Part 3 (Mixer): Each qubit has its own X rotation then its own Y rotation
    for j in range(n):
        qc.rx(2 * betas[j], j)
        qc.ry(2 * alphas[j], j)

    return qc


def cut_values(n, edges):
    """Scores of every assignment, how many interfering pairs are on different channels"""
    values = np.zeros(2**n)
    for index in range(2**n):
        for u, v in edges:
            bit_u = (index >> u) & 1
            bit_v = (index >> v) & 1

            # Different Channel
            if bit_u != bit_v:
                values[index] += 1

    return values


def expected_score(circuit, values):
    """Average score of circuit"""
    probs = Statevector(circuit).probabilities()
    return float(probs @ values)
