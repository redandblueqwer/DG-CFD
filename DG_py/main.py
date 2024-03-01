'''
    1D-scale DG slover

    the struct of slover
    u_h = sum( a_i(t)*phi_i(x) i from 0 to 2)

    Guass_point, Weight = Guass_Legendre_Table(order)
    u(t = 0) = L2_Projection(function)
    get_basis
    Flux
    Lh
    RK
    init_data: define mesh and x
'''

import matplotlib.pyplot as plt
import numpy as np

from Basis_Function import basis_function
from L2_Projection import seg_coef
from RK import rk3

# np.set_printoptions(precision=64)

def inital(x):
    return np.sin(x)


def real_solution(x, t=3.0):
    return np.sin(x - t)


# def Flux: F(U)
# for single wave F(U) = U
def F(x):
    return x


def init_data(a, b, N, u_0, k_order=3):
    """
    :param a: interval begin
    :param b: interval end
    :param N: number of points
    :param u_0: initial value U(x,0)
    :param k_order: the number of Legendre formula (TODO to choose basis_function)
    :return: grid_point
    """

    # 1D-mesh
    delt_x = (b - a) / N
    grid_point = np.arange(a, b + delt_x, delt_x)
    # print(f"grid point coordinates:\n{grid_point}\ndelta_x:{delt_x}\n")

    # init value L2_Projection
    a0 = seg_coef(u_0, a, b, N, k_order)
    # print(f"value of the coefficient at time 0:\n{a0}\n")

    return grid_point, delt_x, a0


def get_plot_point(an_i, x_a, x_b, num=5):
    x_c = (x_a + x_b) / 2.0
    h_half = (x_b - x_a) / 2.0
    x = np.linspace(x_a, x_b, num)
    Xi = (x - x_c) / h_half
    p, _, _ = basis_function(Xi)
    p[0] = np.full(len(p[1]), p[0], dtype=np.float64)
    p = np.vstack(p)
    y = np.dot(an_i, p)
    return x, y


def calculate_L2error(numerical_solution, real_solution):
    diefference = numerical_solution - real_solution
    error = np.sqrt(np.sum(diefference ** 2))
    return error


if __name__ == "__main__":
    # N kinds of grid numbers
    N = np.array([10,20,40])
    total_error = np.zeros(len(N))
    for n in range(0, len(N)):
        tn = 10
        interval, delt_x, a0 = init_data(-2 * np.pi, 2 * np.pi, N[n], inital)
        an = rk3(a0, F, delt_x, delt_t=0.0005, t_end=tn)
        print(f"an shape: {an.shape}")
        # plot
        for i in range(0, len(interval) - 1):
            x, y = get_plot_point(an[i], interval[i], interval[i + 1], num=5)
            plt.scatter(x, y, color='black')
            plt.plot(x, real_solution(x, t=tn))
            total_error[n] = total_error[n] + calculate_L2error(y, real_solution(x, t=tn))

    print(total_error)
    result = np.log2(np.divide(total_error[:-1], total_error[1:]))
    print(result)
    plt.scatter(x, y, label='DG solution', color='black')
    plt.plot(x, real_solution(x, t=tn), label=f'exact solution:sin(x-{tn}) ')
    plt.legend(loc="upper right")
    plt.show()
