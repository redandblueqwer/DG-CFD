import numpy as np
from Guass_Legendre_Table import guass_legendre_table
from Basis_Function import basis_function
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x)

def f2(x):
    u = np.zeros_like(x)
    mask = (x > np.pi/2) & (x < 3*np.pi/2)
    u[mask] = 1
    return u

def projection(f, a, b, order, k_order):
    Guass_point, Weight = guass_legendre_table(order)
    x_c = (a + b) / 2.0
    h = (b - a) / 2.0
    t = Guass_point * h + x_c
    p, M, _ = basis_function(Guass_point, k_order=k_order)
    k_num = len(p)
    coefficient = np.zeros(k_num)
    for i in range(0, k_num):
        integrand = f(t) * p[i] / M[i]
        coefficient[i] = sum(Weight * integrand)
    return coefficient


def seg_coef(f, a, b, N, k_order=3):
    '''
        f      : is continuous function, not include discontinuous solution
        [a,b]  : is total interval
        N      : is segment numbers of interval
        k_order: is the num of basic function numbers, also equal max(Polynomial degree)+1
                 [p0, p1, p2, p3, ..., pm]
        return : coefs of basic function when (t=0)
                 2_dimension array (N,k_order), every row for every interval
    '''

    seg_coefs = np.zeros((N, k_order), dtype=np.float64)
    # print(seg_coefs.shape)
    h = (b - a) / N
    interval = np.arange(a, b + h, h)
    for i in range(0, len(interval) - 1):
        seg_coefs[i, :] = projection(f, interval[i], interval[i + 1], order=7, k_order=k_order)

    return seg_coefs


def plot(function, a, b, num=5):
    x_c = (a + b) / 2.0
    h_half = (b - a) / 2.0
    x = np.linspace(a, b, num)
    t = (x - x_c) / h_half
    coefficient = projection(function, a, b, order=7,k_order=3)
    p, _, _ = basis_function(t,k_order=3)
    y = 0.0
    for i in range(0, len(p)):
        y = y + coefficient[i] * p[i]

    print(coefficient)
    return x, y


if __name__ == "__main__":
    a = 0
    b = 2 * np.pi
    N = 10
    h = (b - a) / (N - 1)
    interval = np.arange(a, b + h, h)
    print(interval)
    coef = seg_coef(f2, a, b, N,k_order=3)
    print(f"coef:{coef}\n")

    for i in range(0, len(interval) - 1):
        x, y = plot(f2, interval[i], interval[i + 1])
        # plt.plot(x, f(x), label='sin exact')
        plt.scatter(x, y, color='black')
        plt.plot(x, f2(x))
    plt.scatter(x, y, label='L2 Projection', color='black')
    # # plt.plot(x,exp(x), label='exp exact')
    plt.plot(x, f2(x), label='sin exact')
    plt.legend()
    plt.show()
