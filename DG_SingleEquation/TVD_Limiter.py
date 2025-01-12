import numpy as np
from Lh import set_bc
from Basis_Function import get_basic_function_data


def minmod(a,b,c,delt_x,uxx):
    if np.abs(a) < delt_x* uxx **2:
        resulta = a
    elif np.all(np.sign(a,b,c)[:-1] == np.sign(a,b,c)[1:]):



def TVD_Limiter(an,k_order):

    an_ghost = set_bc(an, bcL=1, bcR=1)
    data = get_basic_function_data(order=7, k_order=k_order)
    p_begin = data[2]
    p_end = data[3]
    p_center = data[6]
    un_begin = np.dot(an_ghost, p_begin.T)
    un_end = np.dot(an_ghost, p_end.T)
    un_center = np.dot(an_ghost, p_center.T)
    delt_unRC = un_center[2:] - un_center[1:-1]
    delt_unLC = un_center[1:-1] - un_center[0:-2]
    delt_unR =
    delt_unL =