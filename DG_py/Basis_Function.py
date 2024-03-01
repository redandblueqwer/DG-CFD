import numpy as np
from Guass_Legendre_Table import guass_legendre_table

def basis_function(x):
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
    # p = [p0, p1, p2]
    p = [p0, p1, p2, p3]

    p0_x = 0
    p1_x = 1
    p2_x = 2 * x
    p3_x = 3 * x ** 2 - 3 / 5.0
    # p_x = [p0_x, p1_x, p2_x]
    p_x = [p0_x, p1_x, p2_x, p3_x]

    M = [2.0, 2 / 3.0, 8 / 45.0]
    # M = [2.0, 2 / 3.0, 8 / 45.0, 8 / 175.0]

    return p, M, p_x


def get_basic_function_data(order):
    """
        get value of basic function in Guass_point
        return array: p_guass, px_guass, p_begin, p_end, M, Weight
        Be like
         [[p0(x_0),p0(x_1),p0(x_2),p0(x_3),...],
          [p1(x_0),p1(x_1),p1(x_2),p1(x_3),...],
          [p2(x_0),p2(x_1),p2(x_2),p2(x_3),...],
          [p3(x_0),p3(x_1),p3(x_2),p3(x_3),...]]
    :param  order:  Numerical integration order of G-L (order >= GD_order(len(p)))
    """
    Guass_point, Weight = guass_legendre_table(order)
    p_guass, M, px_guass = basis_function(Guass_point)

    p_end, _, _ = basis_function(1)
    p_begin, _, _ = basis_function(-1)

    p_guass[0] = np.full(len(p_guass[1]), p_guass[0],dtype=np.float64)
    px_guass[0] = np.full(len(p_guass[1]), px_guass[0], dtype=np.float64)
    px_guass[1] = np.full(len(p_guass[1]), px_guass[1], dtype=np.float64)
    p_guass = np.vstack(p_guass)
    px_guass = np.vstack(px_guass)
    M = np.array(M)
    Weight = np.array(Weight)
    p_begin = np.array(p_begin)
    p_end = np.array(p_end)

    return p_guass, px_guass, p_begin, p_end, M, Weight


if __name__ == '__main__':
    data = get_basic_function_data(order=7)
    for i in range(0, len(data)):
        print(data[i])
