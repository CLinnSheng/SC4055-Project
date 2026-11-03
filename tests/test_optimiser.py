import pytest

from optimiser import build_circuit

TRIANGLE_EDGES = [(0, 1), (1, 2), (2, 0)]


def test_build_circuit_rejects_too_many_settings():
    with pytest.raises(ValueError):
        build_circuit("xqaoa", [0.0] * 10, 3, TRIANGLE_EDGES)


def test_build_circuit_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_circuit("vqe", [0.0, 0.0], 3, TRIANGLE_EDGES)  # pyright: ignore[reportArgumentType]
