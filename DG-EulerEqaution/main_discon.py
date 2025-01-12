import sys
# 将模块所在的目录添加到sys.path中
sys.path.append('src')

import matplotlib.pyplot as plt
import numpy as np
from IC import IC
from RK import RK3
from Basis_Function import basis_function


def result_plot(An, a, b, grid_point, DG_order, num=5):
    # 可以直接写成向量化运算的形式
    for i in range(0, len(grid_point) - 1):
        x_c = (grid_point[i] + grid_point[i + 1]) / 2.0
        h = (grid_point[i + 1] - grid_point[i]) / 2.0
        x = np.linspace(grid_point[i], grid_point[i + 1], num)
        t = (x - x_c) / h
        p, _, _ = basis_function(t, DG_order)
        y = 0.0
        for k in range(0, len(p)):
            y = y + An[i, k] * p[k]

        plt.scatter(x, y, label='projection')
        # plt.plot(x, y, label='real')
        # plt.plot(x, function(x), label='real')
    plt.show()


N = np.array([10, 100, 20])
DG_order = 2
tn = 0.2
a = -1
b = 1
delt_x = (b - a) / N[1]
points = np.linspace(a, b, N[1] + 1)
An = IC(a, b, N[1], DG_order=DG_order)
print(f"An shape: {An}")

An = RK3(An, delt_x, delt_t=0.0001, t_end=tn)
# print(f"An shape: {An}")
result_plot(An[0], a, b, points, DG_order, num=5)
for i in range(An.shape[0]):
    np.savetxt(f"An_{i}.dat", An[i])
