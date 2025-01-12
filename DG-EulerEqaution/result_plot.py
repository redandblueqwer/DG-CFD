import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from Basis_Function import basis_function


def result_plot(An,grid_point,DG_order,num=5):
    # 可以直接写成向量化运算的形式
    points = []
    rhos = []
    vels = []
    pres = []
    for i in range(0, len(grid_point) - 1): 
        x_c = (grid_point[i] + grid_point[i + 1]) / 2.0
        h = (grid_point[i + 1] - grid_point[i]) / 2.0
        x = np.linspace(grid_point[i], grid_point[i + 1], num)
        t = (x - x_c) / h     
        p, _, _ = basis_function(t, DG_order)
        cfd_data = list([0, 0, 0])  
        for k in range(0, len(p)):
            cfd_data[0] = cfd_data[0] + An[0][i,k] * p[k]
            cfd_data[1] = cfd_data[1] + An[1][i,k] * p[k]
            cfd_data[2] = cfd_data[2] + An[2][i,k] * p[k]
            rho = cfd_data[0]
            vel = cfd_data[1] / cfd_data[0]
            pre = (1.4 - 1) * (cfd_data[2] - 0.5 * cfd_data[1] * vel)
        points.append(x)
        rhos.append(rho)
        vels.append(vel)
        pres.append(pre)

    points = np.array(points).flatten()
    rhos = np.array(rhos).flatten()
    vels = np.array(vels).flatten()
    pres = np.array(pres).flatten()

    # 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制散点图
    ax.scatter(points, rhos, label='rho', color='blue', s=10)
    ax.scatter(points, vels, label='vel', color='red', s=10)
    ax.scatter(points, pres, label='pre', color='black', s=10)
    # ax.plot(points, rhos, label='rho', color='blue', linewidth=5)
    # ax.plot(points, vels, label='vel', color='red', linewidth=5)
    # ax.plot(points, pres, label='pre', color='black', linewidth=5)


    # 设置轴标签
    ax.set_xlabel('x',fontsize=15)
    ax.set_ylabel('y',fontsize=15)

    # 设置轴界限
    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.2, 1.2])

    # 设置标题
    ax.set_title(f'DG-Euler: t=0.2, N=100, DG_order={DG_order}',fontsize=16)

    # 添加图例
    ax.legend(fontsize=14)

    # 设置刻度间隔
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # 设置刻度朝向图内
    # 启用上方和右侧的刻度线，隐藏刻度标签
    ax.tick_params(top=True, right=True, direction='in', length=4, width=1, colors='black', grid_color='black', grid_alpha=0.5, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='both', direction='in')  # 确保所有刻度都朝内

    plt.savefig(f'result-DG{DG_order}.png', format='png', dpi=300)
    # 显示图像
    plt.show()


a = -1
b = 1
N = 100
DG_order = 1
delt_x = (b - a) / N
points = np.linspace(a, b, N+1)

An0 = np.loadtxt(f"Result-DG{DG_order}\An_0.dat")
An1 = np.loadtxt(f"Result-DG{DG_order}\An_1.dat")
An2 = np.loadtxt(f"Result-DG{DG_order}\An_2.dat")
An = np.array([An0, An1, An2])
result_plot(An,points,DG_order,num=10)