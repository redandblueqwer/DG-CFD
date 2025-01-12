'''
step 1: split interval
step 2: save basis function and it's inner product
step 3: compute (f,p_k) and coefficient
step 4: segment projection

'''

import numpy as np
import matplotlib.pyplot as plt
from Guass_Integral import Guass_Legendre_Table

PI = np.pi

# define projection function
def exp(x):
    return np.exp(x)

def sin(x):
    return np.sin(x)

# define basis function and it's inner (pk,pk)
def Legendre(x):
    # LegendreFuction
    p0 = 1
    p1 = x
    p2 = x ** 2 - 1 / 3.0
    p3 = x ** 3 - (3 / 5.0) * x
    p = [p0, p1, p2, p3]

    # 模(pk,pk)
    m0 = 2
    m1 = 2 / 3.0
    m2 = 8 / 45.0
    m3 = 8 / 175.0
    m = [m0, m1, m2, m3]

    return p,m


#  compute the inner product (f,p) in [-1,1]
#  beyond [-1,1] because of (p,p) no equal 0.
def Projection(function,  order):

    point, weight = Guass_Legendre_Table(order)
    p,m = Legendre(point)
    k_num = len(p)
    coefficient = np.zeros(k_num)
    for i in range(0,k_num):
        intergrand = function(point) * p[i] / m[i]
        coefficient[i] = sum(weight * intergrand)

    return coefficient


if __name__ == "__main__":

    coefficient = Projection(sin,order=11)
    print(coefficient)
    x = np.linspace(-1,1,20)
    basic_function,_ = Legendre(x)
    y = 0.0
    for i in range(0,len(basic_function) ):
        y = y + (coefficient[i] * basic_function[i])

    plt.scatter(x,y,label = 'L2 Projection',color = 'black')
    # plt.plot(x,exp(x), label='exp exact')
    plt.plot(x,sin(x),label = 'sin exact')

    plt.legend()
    plt.show()