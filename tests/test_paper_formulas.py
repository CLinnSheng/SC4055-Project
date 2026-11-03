"""
Check the simulator against the paper's exact formulas at random settings.

[1] V. Vijendran et al., arXiv:2302.04479v2 (2024): Eq. (14) for QAOA and
    Theorem 3, Eq. (22) for XQAOA.
"""

import numpy as np
import pytest

from analytic import qaoa_formula, xqaoa_formula
from circuits import cut_values, expected_score, qaoa_circuit, xqaoa_circuit

# Different shapes: no triangles, one triangle, several triangles
NETWORKS = {
    "single_edge": (2, [(0, 1)]),
    "line": (3, [(0, 1), (1, 2)]),
    "triangle": (3, [(0, 1), (1, 2), (2, 0)]),
    "star": (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
    "two_triangles": (
        5,
        [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
    ),  # [1] Fig. 1a
    "prism": (
        6,
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    ),
}
N_RANDOM_SETTINGS = 20


@pytest.mark.parametrize("name", NETWORKS)
def test_qaoa_matches_eq14(name):
    n, edges = NETWORKS[name]
    values = cut_values(n, edges)
    rng = np.random.default_rng(0)

    for _ in range(N_RANDOM_SETTINGS):
        gamma = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(0, np.pi)
        simulated = expected_score(qaoa_circuit(gamma, beta, n, edges), values)

        assert simulated == pytest.approx(qaoa_formula(gamma, beta, n, edges), abs=1e-9)


@pytest.mark.parametrize("name", NETWORKS)
def test_xqaoa_matches_eq22(name):
    n, edges = NETWORKS[name]
    values = cut_values(n, edges)
    rng = np.random.default_rng(1)

    for _ in range(N_RANDOM_SETTINGS):
        gammas = rng.uniform(0, 2 * np.pi, len(edges))
        betas = rng.uniform(0, np.pi, n)
        alphas = rng.uniform(0, np.pi, n)
        simulated = expected_score(
            xqaoa_circuit(gammas, betas, alphas, n, edges), values
        )

        assert simulated == pytest.approx(
            xqaoa_formula(gammas, betas, alphas, n, edges), abs=1e-9
        )
