import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from circuits import cut_values, expected_score, qaoa_circuit, xqaoa_circuit

N_QUBITS = 3
TRIANGLE_EDGES = [(0, 1), (1, 2), (2, 0)]
ZERO_GAMMAS = [0.0] * len(TRIANGLE_EDGES)
ZERO_BETAS = [0.0] * N_QUBITS
ZERO_ALPHAS = [0.0] * N_QUBITS


def test_xqaoa_circuit_has_expected_gates():
    gammas = [0.1, 0.2, 0.3]
    betas = [0.4, 0.5, 0.6]
    alphas = [0.7, 0.8, 0.9]

    circuit = xqaoa_circuit(
        gammas, betas, alphas, N_QUBITS, TRIANGLE_EDGES
    )

    assert circuit.count_ops() == {"h": 3, "rzz": 3, "rx": 3, "ry": 3}


def test_xqaoa_with_zero_angles():
    circuit = xqaoa_circuit(
        ZERO_GAMMAS,
        ZERO_BETAS,
        ZERO_ALPHAS,
        N_QUBITS,
        TRIANGLE_EDGES,
    )
    values = cut_values(N_QUBITS, TRIANGLE_EDGES)

    assert expected_score(circuit, values) == pytest.approx(1.5)


def test_xqaoa_picks_expected_assignment():
    picked_alphas = [np.pi / 4, -np.pi / 4, -np.pi / 4]
    circuit = xqaoa_circuit(
        ZERO_GAMMAS,
        ZERO_BETAS,
        picked_alphas,
        N_QUBITS,
        TRIANGLE_EDGES,
    )
    values = cut_values(N_QUBITS, TRIANGLE_EDGES)

    assert expected_score(circuit, values) == pytest.approx(2.0)


def test_xqaoa_reduces_to_qaoa():
    gamma, beta = 0.5, 0.3
    qaoa = qaoa_circuit(gamma, beta, N_QUBITS, TRIANGLE_EDGES)
    xqaoa = xqaoa_circuit(
        [gamma] * len(TRIANGLE_EDGES),
        [beta] * N_QUBITS,
        ZERO_ALPHAS,
        N_QUBITS,
        TRIANGLE_EDGES,
    )

    assert Statevector(qaoa).equiv(Statevector(xqaoa))


def test_xqaoa_rejects_wrong_number_of_gammas():
    with pytest.raises(ValueError):
        xqaoa_circuit(
            ZERO_GAMMAS[:-1], ZERO_BETAS, ZERO_ALPHAS, N_QUBITS, TRIANGLE_EDGES
        )
