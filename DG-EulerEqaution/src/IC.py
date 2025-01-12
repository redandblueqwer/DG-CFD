from matplotlib import pyplot as plt
import numpy as np
from Basis_Function import basis_function
from L2_Projection import seg_coef


"""
    U1 = rho
    U2 = rho * u
    U3 = rho * e =  0.5 * rho * u^2 + p / (gamma - 1)
"""
def U1_inital(x):
    u = np.zeros_like(x)
    mask = (x >= 0)
    u[mask] = 0.125
    mask = (x < 0)
    u[mask] = 1
    return u

def U2_inital(x):
    u = np.zeros_like(x)
    return u


def U3_inital(x):
    u = np.zeros_like(x)
    mask = (x >= 0)
    u[mask] = 0.1 / 0.4
    mask = (x < 0)
    u[mask] = 1 / 0.4
    return u



def IC(a, b, N, DG_order=3):

    U1 = seg_coef(U1_inital, a, b, N, DG_order)
    U2 = seg_coef(U2_inital, a, b, N, DG_order)
    U3 = seg_coef(U3_inital, a, b, N, DG_order)
    coefficient = np.array([U1, U2, U3])

    return coefficient

def plot_seg(function, a, b, N, DG_order, num=5):

    coefficient = seg_coef(function, a, b, N, DG_order)
    interval = np.linspace(a, b, N + 1)
    # 可以直接写成向量化运算的形式
    for i in range(0, len(interval) - 1): 
        x_c = (interval[i] + interval[i + 1]) / 2.0
        h = (interval[i + 1] - interval[i]) / 2.0
        x = np.linspace(interval[i], interval[i + 1], num)
        t = (x - x_c) / h     
        p, _, _ = basis_function(t, DG_order)
        y = 0.0
        for k in range(0, len(p)):
            y = y + coefficient[i,k] * p[k]

        plt.scatter(x, y, label='projection')
        # plt.plot(x, y, label='real')
        # plt.plot(x, function(x), label='real')
    plt.show()


if __name__ == "__main__":
    a = 0
    b = 1
    N = 10
    coef = IC(a, b, N, DG_order=3)
    plot_seg(U1_inital, a, b, N , DG_order=3, num=5)
    print(f"coef:{coef}\n")
    print(coef.shape)