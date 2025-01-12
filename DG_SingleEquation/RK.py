from Lh import lh


def rk3(an, F, delt_x, delt_t, t_end, k_order):
    """
            U^(1) = U^n + dt * Lh(U^n)
            U^(2) = 3/4 * U^n + 1/4 * U^(1) + 1/4 * dt * Lh(U^(1))
            U^(n+1) = 1/3 * U^n + 2/3 * U^(2) + 2/3 * dt * Lh(U^(2))

    """
    print(f"delta_t:{delt_t}\n")
    t = 0
    while t < t_end:
        if t + delt_t >= t_end:
            delt_t = t_end - t
            t = t_end
            print(t)
        else:
            t = t + delt_t
        # Stage I
        an1 = an + delt_t * lh(an, F, delt_x,k_order=k_order)
        # Stage II
        an2 = (3 / 4.0) * an + (1 / 4.0) * an1 + (1 / 4.0) * delt_t * lh(an1, F, delt_x, k_order=k_order)
        # Stage III
        an = (1 / 3.0) * an + (2 / 3.0) * an2 + (2 / 3.0) * delt_t * lh(an2, F, delt_x, k_order=k_order)
    return an
