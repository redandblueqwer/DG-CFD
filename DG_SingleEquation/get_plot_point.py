import numpy as np
from Basis_Function import basis_function


def get_plot_point(an_i, x_a, x_b, k_order, num=5):
    x_c = (x_a + x_b) / 2.0
    h_half = (x_b - x_a) / 2.0
    x = np.linspace(x_a, x_b, num)
    Xi = (x - x_c) / h_half
    p, _, _ = basis_function(Xi, k_order=k_order)
    p[0] = np.full(len(p[1]), p[0], dtype=np.float64)
    p = np.vstack(p)
    y = np.dot(an_i, p)
    return x, y
