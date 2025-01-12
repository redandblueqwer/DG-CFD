import numpy as np
from Guass_Legendre_Table import guass_legendre_table
from Basis_Function import basis_function,get_basic_function_data
import matplotlib.pyplot as plt


def projection(function, a, b, DG_order):
    Guass_point, _= guass_legendre_table()
    x_c = (a + b) / 2.0
    h = (b - a) / 2.0
    t = Guass_point * h + x_c
    p_guass,_ ,_ ,_ , M, Weight ,_ = get_basic_function_data(DG_order)
    coefficient = np.zeros_like(M)
    integrand = np.zeros_like(M)
    integrand = np.dot(function(t) * p_guass, Weight.T) # 高斯积分
    
    coefficient = integrand / M
    return coefficient


def seg_coef(function, a, b, N, DG_order):
    '''
        function : function
        [a,b]  : is total interval
        N      : is segment numbers of interval
        DG_order: is  max(Polynomial degree), also equal the num of basic function numbers - 1
        return : coefs of basic function when (t=0)
                 2_dimension array (N,k_order), every row for every interval
    '''
    coefficient = np.zeros((N, DG_order + 1), dtype=np.float64)
    interval = np.linspace(a, b, N + 1)
    Guass_point, _ = guass_legendre_table()
    p_guass,_ ,_ ,_ , M, Weight ,_ = get_basic_function_data(DG_order)
    for i in range(0, len(interval) - 1): 
        x_c = (interval[i] + interval[i + 1]) / 2.0
        h = (interval[i + 1] - interval[i]) / 2.0
        t = Guass_point * h + x_c     
        integrand = np.dot(function(t) * p_guass, Weight.T) # 高斯积分
        coefficient[i] = integrand / M

    return coefficient



# test part

def f(x):
    return np.sin(x)

def f2(x):
    u = np.zeros_like(x)
    mask = (x > np.pi/2) & (x < 3*np.pi/2)
    u[mask] = 1
    return u



def plot(function, a, b, num=10):
    x_c = (a + b) / 2.0
    h = (b - a) / 2.0
    x = np.linspace(a, b, num)
    t = (x - x_c) / h
    DG_order = 3
    coefficient = projection(function, a, b, DG_order)
    p, _, _ = basis_function(t, DG_order)
    y = 0.0
    for i in range(0, len(p)):
        y = y + coefficient[i] * p[i]

    plt.plot(x, y, label='projection')
    plt.plot(x, function(x), label='real')
    plt.show()

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
        #plt.plot(x, y, label='real')
        # plt.plot(x, function(x), label='real')
    plt.show()





if __name__ == "__main__":
    a = 0
    b = 2 * np.pi
    N = 10
    plot(f, a, b,num=100)
    plot_seg(f2, a, b, N, DG_order=3, num=5)


   
