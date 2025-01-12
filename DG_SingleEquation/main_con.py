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
from get_plot_point import get_plot_point
from init_data import init_data
from RK import rk3
from calulate_L2error import calculate_L2error


def inital(x):
    return np.sin(x)


def real_solution(x, tn):
    return np.sin(x - tn)


# To all Matrix operations should use the symbol function
def f_real_solution(x):
    return np.sin(x - 2.0)


# def Flux: F(U)
# for single wave F(U) = U
def F(x):
    return x


if __name__ == "__main__":
    # N kinds of grid numbers
    N = np.array([20])
    total_error = np.zeros(len(N))
    k_order = 4
    tn = 2.0
    for n in range(0, len(N)):
        interval, delt_x, a0 = init_data(-2 * np.pi, 2 * np.pi, N[n], inital, k_order=k_order)
        an = rk3(a0, F, delt_x, delt_t=0.05, t_end=tn, k_order=k_order)
        print(f"an shape: {an.shape}")
        # error: [3.50028064 3.47780967]
        for i in range(0, len(an)):
            total_error[n] = total_error[n] + calculate_L2error(an[i], f_real_solution, interval[i], interval[i + 1],
                                                                k_order=k_order)
    print(total_error)
    print(np.log2(np.divide(total_error[:-1], total_error[1:])))

    # plot
    interval, delt_x, a0 = init_data(-2 * np.pi, 2 * np.pi, N[-1], inital, k_order=k_order)
    an = rk3(a0, F, delt_x, delt_t=0.005, t_end=tn, k_order=k_order)
    for i in range(0, len(interval) - 1):
        x, y = get_plot_point(an[i], interval[i], interval[i + 1], k_order=k_order, num=2)
        plt.scatter(x, y, color='black')
        plt.plot(x, real_solution(x, tn))

    plt.scatter(x, y, label='DG solution', color='black')
    # plt.plot(x, real_solution(x, tn), label=f'exact solution:sin(x-{tn})')
    plt.legend(loc="upper right")
    plt.show()
