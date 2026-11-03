"Independent exact-answer and scoring checks, using pytest like the quantum tests."
from itertools import product
import pytest
from classical_reference import NETWORKS, cut_score, solve

EXPECTED = {
    "single_edge": (1, 2),
    "line": (2, 2),
    "triangle": (2, 6),
    "star": (4, 2),
    "two_triangles": (4, 18),
    "prism": (7, 6),
}


def test_single_edge_scores():
    assert [cut_score(bits, [(0, 1)]) for bits in product((0, 1), repeat=2)] == [0, 1, 1, 0]


@pytest.mark.parametrize("name", NETWORKS)
def test_exact_answers(name):
    n, edges = NETWORKS[name]
    result = solve(n, edges)
    optimum, count = EXPECTED[name]
    assert result["optimal_score"] == optimum
    assert result["n_optimal_assignments"] == count
    assert result["assignments_checked"] == 2**n
    assert len({row["assignment"] for row in result["all_scores"]}) == 2**n


@pytest.mark.parametrize("name", NETWORKS)
def test_complements_and_random_baseline(name):
    n, edges = NETWORKS[name]
    for bits in product((0, 1), repeat=n):
        assert cut_score(bits, edges) == cut_score(tuple(1-b for b in bits), edges)
    assert solve(n, edges)["random_expected_score"] == pytest.approx(len(edges) / 2)
    assert cut_score((0,) * n, edges) == 0


def test_star_optimal_assignments():
    n, edges = NETWORKS["star"]
    assert solve(n, edges)["optimal_assignments"] == ["01111", "10000"]


def test_duplicate_reversed_edge_rejected():
    with pytest.raises(ValueError):
        solve(2, [(0, 1), (1, 0)])
