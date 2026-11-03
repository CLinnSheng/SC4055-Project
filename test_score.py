import numpy as np

from circuits import cut_values, expected_score, qaoa_circuit

# Test 1: one pair of transmitters that interfere
single_edge = [(0, 1)]
single_edge_values = cut_values(2, single_edge)
print("scores of 00, 10, 01, 11:", single_edge_values)

zero_circuit = qaoa_circuit(0, 0, 2, single_edge)
print("dials at zero:", expected_score(zero_circuit, single_edge_values))

best_circuit = qaoa_circuit(np.pi / 2, np.pi / 8, 2, single_edge)
print("best dials:  ", expected_score(best_circuit, single_edge_values))

# Test 2: triangle
triangle_edges = [(0, 1), (1, 2), (2, 0)]
triangle_values = cut_values(3, triangle_edges)
print("triangle scores:", triangle_values)

zero_circuit = qaoa_circuit(0, 0, 3, triangle_edges)
print("dials at zero:", expected_score(zero_circuit, triangle_values))
