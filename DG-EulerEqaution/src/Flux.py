import numpy as np
gamma = 1.4

"""
    define numerical flux
    flux.shape = [n_equation = 3, Ghost_An.shape[1] - 1]
"""

def original_data(Un):
    """
    :param Un:  Un = [U1, U2, U3]
    :return:  rho, u, p
    """
    rho = Un[0]
    u = Un[1] / Un[0]
    p = (gamma - 1) * (Un[2] - 0.5 * Un[1] * Un[1] / Un[0])
    return rho, u, p


def lax_friedrichs_flux(Flux_L, Flux_R, Un_L, Un_R):
    
    rho_L, u_L, p_L = original_data(Un_L)
    rho_R, u_R, p_R = original_data(Un_R)
    # print(rho_R)
    c_L = np.sqrt(np.abs(gamma * p_L / rho_L))
    c_R = np.sqrt(np.abs(gamma * p_R / rho_R))
    c = np.maximum(c_L, c_R)
    alpha = np.maximum(np.abs(u_L) + c, np.abs(u_R) + c)
    # print(f"alpha:{alpha}")
    Flux = 0.5 * (Flux_L + Flux_R) - 0.5 * alpha * (Un_R - Un_L)

    return Flux


def hll():

    return  0

def hllc():

    return 0
