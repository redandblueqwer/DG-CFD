import numpy as np
from L2_Projection import seg_coef


def init_data(a, b, N, u_0, k_order=3):
    """
    :param a: interval begin
    :param b: interval end
    :param N: number of points
    :param u_0: initial value U(x,0)
    :param k_order: the number of Legendre formula ( to choose basis_function)
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