"""
The project writes every assignment with node 0 first (left to right),
the same as classical_reference.py. These tests check that the quantum side
(circuits.py, Qiskit order) and the classical side agree.
"""

import pytest

from circuits import cut_values, from_qiskit_bitstring, to_assignment
from classical_reference import NETWORKS, cut_score, solve


def test_to_assignment_puts_node_0_first():
    # index 1 = binary 001 in Qiskit order, so only node 0 is on channel 1
    assert to_assignment(1, 3) == "100"
    assert to_assignment(6, 3) == "011"


def test_from_qiskit_bitstring_reverses():
    assert from_qiskit_bitstring("001") == "100"


@pytest.mark.parametrize("name", NETWORKS)
def test_quantum_and_classical_scores_agree(name):
    n, edges = NETWORKS[name]
    values = cut_values(n, edges)

    for index in range(2**n):
        assignment = [int(c) for c in to_assignment(index, n)]
        assert values[index] == cut_score(assignment, edges)


@pytest.mark.parametrize("name", NETWORKS)
def test_quantum_and_classical_optimal_assignments_agree(name):
    n, edges = NETWORKS[name]
    values = cut_values(n, edges)

    best = values.max()
    quantum_side = {to_assignment(i, n) for i in range(2**n) if values[i] == best}

    assert quantum_side == set(solve(n, edges)["optimal_assignments"])
