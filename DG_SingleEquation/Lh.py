import numpy as np
from Basis_Function import get_basic_function_data
from Flux import lax_friedrichs_flux


def set_bc(an, bcL, bcR):
    ghost = np.zeros((1, an.shape[1]))
    an_ghost = np.concatenate((ghost, an, ghost), axis=0)
    # set bc
    if bcL == 1:
        an_ghost[0, :] = an[-1, :]

    if bcR == 1:
        an_ghost[-1, :] = an[0, :]

    return an_ghost


def lh(an, F, delt_x, k_order = 3):
    """
    :param
             an: time coefficient at time n.
    :return:
             result: spatial discrete values at time n.

    """
    data = get_basic_function_data(order=7, k_order=k_order)
    p_guass = data[0]
    px_guass = data[1]
    p_begin = data[2]
    p_end = data[3]
    M = data[4]
    weight = data[5]
    h = delt_x / 2.0
    interval_num = an.shape[0]
    # print(interval_num)
    an_ghost = set_bc(an, bcL=1, bcR=1)

    # step1 calculate the Integral in ceil
    # input is an.shape, ouput also is an.shape
    un = np.dot(an, p_guass)
    # print(un.shape)
    Integral_result = np.dot(weight * F(un), px_guass.T)

    # step2 calculate the flux at edge

    # compute value of uh at interval point
    un_begin = np.dot(an_ghost, p_begin.T)
    un_end = np.dot(an_ghost, p_end.T)
    # print(f"p_end is {p_end.shape} \n")
    # print(f"un_begin is {un_begin.shape} \n")
    point_number = interval_num + 1
    flux = np.zeros(point_number)
    for i in range(0, point_number):
        flux[i] = lax_friedrichs_flux(F, un_end[i], un_begin[i + 1], alpha=1)
    
    print(f"flux is{flux} \n")

    Boun_result = Integral_result.copy()
    for i in range(0, interval_num):
        
        Boun_result[i] = flux[i + 1] * p_end - flux[i] * p_begin
    
    # print(f"Boun_result is {Boun_result} \n")
    # step3 calculate the sum of Lh(un) in all cell [a,b]
    lh_result = (Integral_result - Boun_result) / (M * h)
    # print(f"lh_result is {lh_result.shape} \n")
    return lh_result


def test(x):
    return 1 * x


if __name__ == "__main__":
    an = np.array([[0.4597, 0.4279, -0.0589, -0.0180],
                   [1, 2, 3, 4]])
    print(an)

    re = 0.001 * lh(an, test, 0.25,k_order=4)
    print(re)
