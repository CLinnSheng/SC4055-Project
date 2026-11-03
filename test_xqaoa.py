import numpy as np
from qiskit.quantum_info import Statevector

from circuits import cut_values, expected_score, qaoa_circuit, xqaoa_circuit

# triangle
n = 3
edges = [(0, 1), (1, 2), (2, 0)]

gamma, beta = 0.5, 0.3

gammas = [0.1, 0.2, 0.3]
betas = [0.4, 0.5, 0.6]
alphas = [0.7, 0.8, 0.9]

x = xqaoa_circuit(gammas, betas, alphas, n, edges)
print(x.draw())
print(x.count_ops())

# With all dials at zero, XQAOA is only H gates, so it is a random guess
#    Triangle: average of the 8 scores = (0 + 2*6 + 0) / 8 = 1.5
zero_gammas = [0.0] * len(edges)
zero_betas = [0.0] * n
zero_alphas = [0.0] * n

x_zero = xqaoa_circuit(zero_gammas, zero_betas, zero_alphas, n, edges)
print("Score with dials at zero:", expected_score(x_zero, cut_values(n, edges)))

#    gamma = 0, beta = 0, alpha = +pi/4 puts a qubit on 1 and -pi/4 puts it on 0
#    Assignment 001 separates 2 of the triangle's 3 pairs, the best possible
picked_alphas = [np.pi / 4, -np.pi / 4, -np.pi / 4]
x_pick = xqaoa_circuit(zero_gammas, zero_betas, picked_alphas, n, edges)
print("Score of picked assignment:", expected_score(x_pick, cut_values(n, edges)))

# 4. Same state as our QAOA circuit when all gammas are equal, all betas are equal and all alphas = 0
q = qaoa_circuit(gamma, beta, n, edges)
x_as_q = xqaoa_circuit(
    [gamma] * len(edges),
    [beta] * n,
    zero_alphas,
    n,
    edges,
)
print("Reduces to QAOA:", Statevector(q).equiv(Statevector(x_as_q)))
