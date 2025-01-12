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


def exp(x):
    return np.exp(x)
def sin(x):
    return np.sin(x)

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

#  x in [a,b]  and  t in [-1,1]
#  x = (b-a)/2 * t + (a+b)/2
#  compute inner product (F(t),p(t)), then F(t) = f((b-a)/2 * t + (a+b)/2)
def Projection(function,a,b,order):
    point, weight = Guass_Legendre_Table(order)
    h = (b - a) / 2.0
    c = (a + b) / 2.0
    x = h * point + c  # computer real point x

    p,m = Legendre(point)
    k_num = len(p)
    coefficient = np.zeros(k_num)
    for i in range(0,k_num):
        intergrand = function(x) * p[i] / m[i]
        coefficient[i] = sum(weight * intergrand)

    return coefficient

def SegmentProjection(function,a,b,num):
    # a,b 为区间起点与终点
    # num [a,b]区间取的点数目
    # y   返回[a,b]区间上对应的投影值
    h = (b - a) / 2.0
    c = (a + b) / 2.0
    x = np.linspace(a, b, num)
    t = (x - c) / h
    coefficient = Projection(function, a, b, order=11)

    basic_function, _ = Legendre(t)
    y = 0.0
    for i in range(0, len(basic_function)):
        y = y + (coefficient[i] * basic_function[i])
    return x,y

if __name__ == "__main__":
    # split [a,b] in N parts, then save to interval
    a = -10.0
    b = 10.0
    N = 10
    h = (b - a) / N
    interval = np.arange(a,b+1,h)
    print(interval)

    for i in range(0,len(interval)-1):
        x,y = SegmentProjection(sin,interval[i],interval[i+1],10)
        plt.scatter(x,y,color = 'black')

    plt.scatter(x, y,label = 'L2 Projection', color='black')
    x = np.linspace(a, b, 100)
    # plt.plot(x,exp(x), label='exp exact')
    plt.plot(x, sin(x), label='sin exact')
    plt.legend()
    plt.show()