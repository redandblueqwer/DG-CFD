import numpy as np
from Basis_Function import basis_function
from Guass_Legendre_Table import guass_legendre_table


def sin(x):
    return np.sin(x)


def calculate_L2error(an_i, f, x_a, x_b, k_order, order=7):
    Guass_point, Weight = guass_legendre_table(order)
    p_guass, _, _ = basis_function(Guass_point, k_order=k_order)

    p_guass[0] = np.full(len(p_guass[1]), p_guass[0], dtype=np.float64)
    p_guass = np.vstack(p_guass)
    Weight = np.array(Weight)
    Guass_point = np.array(Guass_point)
    x_c = (x_a + x_b) / 2.0
    h_half = (x_b - x_a) / 2.0
    Xi = x_c + h_half * Guass_point
    # 做高斯积分
    numerical_solution = np.dot(an_i, p_guass)
    real_solution = f(Xi)
    error = np.dot(np.abs(numerical_solution - real_solution) ** 2, (h_half * Weight).T)
    error = np.sqrt(np.sum(error))
    return error


if __name__ == '__main__':
    an = np.array([[0.4597, 0.4279, -0.0589, -0.0180],
                   [0, 0, 0, 0],
                   [1, 2, 3, 4]])
    ans = calculate_L2error(an, sin, x_a=1, x_b=2, k_order=4, order=7)
    print(ans)
