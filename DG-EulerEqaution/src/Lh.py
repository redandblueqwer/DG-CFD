import numpy as np
from Basis_Function import get_basic_function_data
from Flux import lax_friedrichs_flux


gamma = 1.4
n_equation = 3

# 标量方程(1个方程)
# An = np.array(n_equation = 3, interval_num, DG_order + 1)
def set_bc(An, bcL, bcR):
    
    An_ghost = np.zeros((n_equation, An.shape[1] + 2, An.shape[2]))
    An_ghost[:, 1:-1, :] = An
    for i in range(0, n_equation):
        # set period bc
        if bcL == 1:
            An_ghost[i, 0, :] = An[i, -1, :]
        if bcR == 1:
            An_ghost[i, -1, :] = An[i, 0, :]
    
        # set reflect bc
        if bcL == 2:
            An_ghost[i, 0, :] = An[i, 0, :]
        if bcR == 2:
            An_ghost[i, -1, :] = An[i, -1, :]

    return An_ghost

def Invis_Flux(Un):
    Flux = np.zeros_like(Un)
    Flux[0] = Un[1]
    Flux[1] = 0.5 * (3 - gamma) * Un[1] ** 2 / Un[0] + (gamma - 1) * Un[2]
    Flux[2] = 0.5 * (1 - gamma) * Un[1] ** 3 / Un[0] + gamma * Un[1] * Un[2] / Un[0] 
    return Flux

def Lh(An, delt_x, DG_order):
    """
    :param
             An: time coefficient at time n.
    :return:
             result: spatial discrete values at time n.

    """


    data = get_basic_function_data( DG_order = DG_order)
    p_guass = data[0]
    px_guass = data[1]
    p_left = data[2]
    p_right = data[3]
    M = data[4]
    weight = data[5]
    h = delt_x / 2.0
    interval_num = An.shape[1]      # 分割的区间数 
    An_ghost = set_bc(An, bcL=2, bcR=2)

    #  p_guass.shape[1] = 5 
    #  Intergal_order = 9 使用5个Gauss点
    Un = np.zeros((n_equation, An.shape[1], p_guass.shape[1]))
    Un_L = np.zeros((n_equation, An_ghost.shape[1]))
    Un_R = np.zeros((n_equation, An_ghost.shape[1]))
 

    for i in range(0, n_equation):
        Un[i] = np.dot(An[i], p_guass)
        # Left and Right 对应来说端点, [-1,1]相对于区间来说
        # 则 Un_L 为 x=1 处的值, Un_R 为 x=-1  处的值
        Un_L[i] = np.dot(An_ghost[i], p_left.T)
        Un_R[i] = np.dot(An_ghost[i], p_right.T)

    Flux = Invis_Flux(Un)
    Flux_L = Invis_Flux(Un_L)
    Flux_R = Invis_Flux(Un_R)

    Integral_result_1 = np.dot(Flux[0] , (px_guass * weight).T)
    Integral_result_2 = np.dot(Flux[1] , (px_guass * weight).T)
    Integral_result_3 = np.dot(Flux[2] , (px_guass * weight).T)
    Integral_result = np.array([Integral_result_1, Integral_result_2, Integral_result_3])

    # print(f"Un_L is {Un_L} \n")
    flux =  np.zeros((n_equation, interval_num + 1))
    for i in range(0, interval_num + 1):
        flux[:,i] = lax_friedrichs_flux(Flux_L[:,i], Flux_R[:,i+1], Un_L[:,i], Un_R[:,i+1])
    # print(f"flux is {flux} \n")
    # step2 calculate the flux at point
    Boun_result = Integral_result.copy()
    for i in range(0, n_equation):
        for j in range(0, interval_num):
            Boun_result[i,j] = flux[i,  j + 1] * p_left - flux[i,  j] * p_right
 
    # print(Boun_result.shape)

    Lh_result = (Integral_result - Boun_result) / (M * h)
    # print(f"lh_result is {Lh_result.shape} \n")
    return Lh_result




if __name__ == "__main__":
    An_test = np.array([[[1.0, 0.0, 0.0, 0.0],
                  [0.438055556, -0.620058099, 0.466666667, 0.620640789],
                  [0.125, 0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0, 0.0],
                  [0.0, -0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]],
                 [[2.5, 0.0, 0.0, 0.0],
                  [1.055, -1.59443511, 1.2, 1.59593346],
                  [0.25, 0.0, -6.9388939e-17, 0.0]]])



    An_ghost = np.zeros((n_equation, An_test.shape[1] + 2, An_test.shape[2]))
    An_ghost[:, 1:-1, :] = An_test
    # print(An_ghost.shape)

    re = 0.001 * Lh(An_test, 0.25, DG_order=3)
    # print(re)
