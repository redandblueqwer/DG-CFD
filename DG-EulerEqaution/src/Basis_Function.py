import numpy as np
from Guass_Legendre_Table import guass_legendre_table


def basis_function(x, DG_order=2):
    """
        basis_function :               p = [p0, p1, p2, p3]
        partial_derivations function : p_x = [p0_x, p1_x, p2_x, p3_x]
        inner product at [-1,1] :      M = [2, 2 / 3.0, 8 / 45.0, 8 / 175.0]
        Tode write 递推式
    """
    p0 = 1
    p1 = x
    p2 = x ** 2 - 1 / 3.0
    p3 = x ** 3 - (3 / 5.0) * x
    p4 = x ** 4 - (30 / 35.0) * x ** 2 + 3 / 35.0
    p5 = x ** 5 - (70 / 63.0) * x ** 3 + 15 / 63.0 * x

    p0_x = 0
    p1_x = 1
    p2_x = 2 * x
    p3_x = 3 * x ** 2 - 3 / 5.0
    p4_x = 4 * x ** 3 - (60 / 35.0) * x
    p5_x = 5 * x ** 4 - (210 / 63.0) * x ** 2 + 15 / 63.0

    M0 = 2.0
    M1 = 2 / 3.0
    M2 = 8 / 45.0
    M3 = 8 / 175.0
    M4 = 128 / 11025
    M5 = 3.017929075 * 10E-3

    if DG_order == 1:
        p = [p0, p1]
        p_x = [p0_x, p1_x]
        M = [M0, M1]

    if DG_order == 2:
        p = [p0, p1, p2]
        p_x = [p0_x, p1_x, p2_x]
        M = [M0, M1, M2]
    if DG_order == 3:
        p = [p0, p1, p2, p3]
        p_x = [p0_x, p1_x, p2_x, p3_x]
        M = [M0, M1, M2, M3]
    if DG_order == 4:
        p = [p0, p1, p2, p3, p4]
        p_x = [p0_x, p1_x, p2_x, p3_x, p4_x]
        M = [M0, M1, M2, M3, M4]
    if DG_order == 5:
        p = [p0, p1, p2, p3, p4, p5]
        p_x = [p0_x, p1_x, p2_x, p3_x, p4_x, p5_x]
        M = [M0, M1, M2, M3, M4, M5]


    return p, M, p_x


def get_basic_function_data( DG_order):
    """
        Get Value of Basic Function in Guass_point
        return array: p_guass, px_guass, p_begin, p_end, M, Weight
        Be like
         [[p0(x_0),p0(x_1),p0(x_2),p0(x_3),...],
          [p1(x_0),p1(x_1),p1(x_2),p1(x_3),...],
          [p2(x_0),p2(x_1),p2(x_2),p2(x_3),...],
          [p3(x_0),p3(x_1),p3(x_2),p3(x_3),...]]
    :param k_order:
    :param  order:  Numerical integration order of G-L (order >= GD_order(len(p)))
    """
    Guass_point, Weight = guass_legendre_table()
    p_guass, M, px_guass = basis_function(Guass_point, DG_order)
    p_left, _, _ = basis_function(x=1, DG_order = DG_order)
    p_right, _, _ = basis_function(x=-1, DG_order= DG_order)
    p_center, _, _ = basis_function(x=0, DG_order= DG_order)


    p_guass[0] = np.full(len(p_guass[1]), p_guass[0], dtype=np.float64)
    px_guass[0] = np.full(len(p_guass[1]), px_guass[0], dtype=np.float64)
    px_guass[1] = np.full(len(p_guass[1]), px_guass[1], dtype=np.float64)
    p_guass = np.vstack(p_guass)
    px_guass = np.vstack(px_guass)
    M = np.array(M)
    Weight = np.array(Weight)
    p_left = np.array(p_left)
    p_right = np.array(p_right)
    p_center = np.array(p_center)

    return p_guass, px_guass, p_left, p_right, M, Weight, p_center


if __name__ == '__main__':
    data = get_basic_function_data(DG_order=3)
    for i in range(0, len(data)):
        print(data[i])
