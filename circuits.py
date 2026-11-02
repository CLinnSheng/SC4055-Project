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


def cut_values(n, edges):
    """Scores of every assignment, how many interfering paris are on different channels"""
    values = np.zeros(2**n)
    for index in range(2**n):
        for u, v in edges:
            bit_u = (index >> u) & 1
            bit_v = (index >> v) & 1

            # Different Channel
            if bit_u != bit_v:
                values[index] += 1

    return values

def expected_scores(circuit, values):
    """Average score of circuit"""
    probs = Statevector(circuit).probabilities()
    return float(probs * values)
