import numpy as np
import pytest

from circuits import cut_values, expected_score, qaoa_circuit


def test_qaoa_circuit_has_expected_gates():
    edges = [(0, 1), (1, 2), (2, 0)]

    circuit = qaoa_circuit(0.5, 0.3, 3, edges)

    assert circuit.count_ops() == {"h": 3, "rzz": 3, "rx": 3}


def test_single_edge_cut_values():
    values = cut_values(2, [(0, 1)])

    assert values.tolist() == [0, 1, 1, 0]


def test_triangle_cut_values():
    edges = [(0, 1), (1, 2), (2, 0)]

    values = cut_values(3, edges)

    assert values.tolist() == [0, 2, 2, 2, 2, 2, 2, 0]


def test_single_edge_with_zero_angles():
    edges = [(0, 1)]
    circuit = qaoa_circuit(0, 0, 2, edges)
    values = cut_values(2, edges)

    assert expected_score(circuit, values) == pytest.approx(0.5)


def test_triangle_with_zero_angles():
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = qaoa_circuit(0, 0, 3, edges)
    values = cut_values(3, edges)

    assert expected_score(circuit, values) == pytest.approx(1.5)


def test_single_edge_with_best_angles():
    edges = [(0, 1)]
    circuit = qaoa_circuit(np.pi / 2, np.pi / 8, 2, edges)
    values = cut_values(2, edges)

    assert expected_score(circuit, values) == pytest.approx(1.0)
