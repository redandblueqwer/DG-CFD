from Lh import Lh


def RK3(An, delt_x, delt_t, t_end):
    """
            U^(1) = U^n + dt * Lh(U^n)
            U^(2) = 3/4 * U^n + 1/4 * U^(1) + 1/4 * dt * Lh(U^(1))
            U^(n+1) = 1/3 * U^n + 2/3 * U^(2) + 2/3 * dt * Lh(U^(2))

    """
    print(f"delta_t:{delt_t}\n")
    DG_order = An.shape[2] - 1
    t = 0
    while t < t_end:
        if t + delt_t >= t_end:
            delt_t = t_end - t
            t = t_end
            print(t)
        else:
            t = t + delt_t
        # Stage I
        An1 = An + delt_t * Lh(An, delt_x, DG_order)
        # print(f"An1 shape: {An1.shape}")
        # Stage II
        An2 = (3 / 4.0) * An + (1 / 4.0) * An1 + (1 / 4.0) * delt_t * Lh(An1, delt_x, DG_order)
        # Stage III
        An = (1 / 3.0) * An + (2 / 3.0) * An2 + (2 / 3.0) * delt_t * Lh(An2, delt_x, DG_order)
    return An
