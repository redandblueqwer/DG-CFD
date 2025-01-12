
"""
    define numerical flux
"""
def lax_friedrichs_flux(f, u_l, u_r, alpha=1):
    flux = 0.5 * (f(u_l) + f(u_r) + alpha * (u_l - u_r))

    return flux

def hll():

    return  0

def hllc():

    return 0
