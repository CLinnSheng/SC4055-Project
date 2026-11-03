import numpy as np

from circuits import cut_values, expected_score, qaoa_circuit

n = 3
triangle_edges = [(0, 1), (1, 2), (2, 0)]
gamma, beta = 0.5, 0.3

circuit = qaoa_circuit(gamma, beta, n, triangle_edges)
print(circuit.draw())
print(circuit.count_ops())

# With all dials at zero, QAOA is only H gates, so it is a random guess
#    Triangle: average of the 8 scores = (0 + 2*6 + 0) / 8 = 1.5
zero_circuit = qaoa_circuit(0, 0, n, triangle_edges)
triangle_values = cut_values(n, triangle_edges)
print(
    "Score with dials at zero:",
    expected_score(zero_circuit, triangle_values),
)

# Single edge (T0 — T1): with no neighbours, the paper's eq. (14) simplifies to
#    1/2 + 1/2 * sin(4*beta) * sin(gamma), which reaches the perfect score 1
#    at gamma = pi/2 and beta = pi/8
single_edge = [(0, 1)]
best_circuit = qaoa_circuit(np.pi / 2, np.pi / 8, 2, single_edge)
single_edge_values = cut_values(2, single_edge)
print(
    "Single edge, best dials:",
    expected_score(best_circuit, single_edge_values),
)
