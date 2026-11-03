"""
Exact one-layer formulas from the paper, used ONLY to check the simulator.
Unweighted graphs (every w_uv = 1).

References
----------
[1] V. Vijendran et al., "An Expressive Ansatz for Low-Depth Quantum
    Approximate Optimisation", arXiv:2302.04479v2 (2024).

Notation for an edge u-v, [1] Sec. II B:
    e = neighbours of v other than u
    d = neighbours of u other than v
    F = neighbours of both u and v (they form triangles with the edge)
"""

import numpy as np

from circuits import Angles, Edges


def neighbours(n: int, edges: Edges) -> dict[int, set[int]]:
    """Which transmitters each transmitter is connected to."""
    nbrs: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        nbrs[u].add(v)
        nbrs[v].add(u)
    return nbrs


def qaoa_formula(gamma: float, beta: float, n: int, edges: Edges) -> float:
    """Average score of one-layer QAOA, [1] Eq. (14), summed over all edges
    ([1] Theorem 1)."""
    nbrs = neighbours(n, edges)
    total = 0.0
    for u, v in edges:
        e = nbrs[v] - {u}  # other neighbours of v
        d = nbrs[u] - {v}  # other neighbours of u
        F = nbrs[u] & nbrs[v]  # shared neighbours (triangles)

        total += 0.5 + 0.25 * (
            np.sin(4 * beta)
            * np.sin(gamma)
            * (np.cos(gamma) ** len(e) + np.cos(gamma) ** len(d))
            + np.sin(2 * beta) ** 2
            * np.cos(gamma) ** (len(e) + len(d) - 2 * len(F))
            * (np.cos(2 * gamma) ** len(F) - 1)
        )
    return float(total)


def xqaoa_formula(
    gammas: Angles, betas: Angles, alphas: Angles, n: int, edges: Edges
) -> float:
    """Average score of one-layer XQAOA with the XY mixer, [1] Theorem 3,
    Eq. (22), summed over all edges."""
    nbrs = neighbours(n, edges)

    # gamma of each edge, looked up in either direction: g[(u, v)] == g[(v, u)]
    g: dict[tuple[int, int], float] = {}
    for (u, v), angle in zip(edges, gammas):
        g[(u, v)] = g[(v, u)] = angle

    total = 0.0
    for u, v in edges:
        e = nbrs[v] - {u}  # other neighbours of v
        d = nbrs[u] - {v}  # other neighbours of u
        F = nbrs[u] & nbrs[v]  # shared neighbours (triangles)
        au, av, bu, bv = alphas[u], alphas[v], betas[u], betas[v]

        # Products of cosines in Eq. (22)
        prod_e = np.prod([np.cos(g[(w, v)]) for w in e])
        prod_d = np.prod([np.cos(g[(u, w)]) for w in d])
        prod_e_not_F = np.prod([np.cos(g[(w, v)]) for w in e - F])  # w in e, w not in F
        prod_d_not_F = np.prod([np.cos(g[(u, w)]) for w in d - F])  # w in d, w not in F
        tri_plus = np.prod([np.cos(g[(u, f)] + g[(v, f)]) for f in F])
        tri_minus = np.prod([np.cos(g[(u, f)] - g[(v, f)]) for f in F])

        # Line 1 of Eq. (22)
        term1 = (
            np.cos(2 * au)
            * np.cos(2 * av)
            * np.sin(g[(u, v)])
            * (
                np.cos(2 * bu) * np.sin(2 * bv) * prod_e
                + np.sin(2 * bu) * np.cos(2 * bv) * prod_d
            )
        )
        # Line 2 of Eq. (22)
        term2 = (
            -0.5
            * np.sin(2 * au)
            * np.sin(2 * av)
            * prod_e_not_F
            * prod_d_not_F
            * (tri_plus + tri_minus)
        )
        # Line 3 of Eq. (22)
        term3 = (
            0.5
            * np.cos(2 * au)
            * np.sin(2 * bu)
            * np.cos(2 * av)
            * np.sin(2 * bv)
            * prod_e_not_F
            * prod_d_not_F
            * (tri_plus - tri_minus)
        )

        total += 0.5 + 0.5 * (term1 + term2 + term3)
    return float(total)
