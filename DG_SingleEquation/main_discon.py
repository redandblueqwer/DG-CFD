import matplotlib.pyplot as plt
import numpy as np
from get_plot_point import get_plot_point
from init_data import init_data
from RK import rk3




def inital(x):
    u = np.zeros_like(x)
    mask = (x > np.pi/2) & (x < 3*np.pi/2)
    u[mask] = 1
    return u

def F(x):
    return x


N = np.array([10,20])
k_order = 5
tn = 2*np.pi
interval, delt_x, a0 = init_data(0, 2 * np.pi, N[1], inital, k_order=k_order)
an = rk3(a0, F, delt_x, delt_t=0.005, t_end=tn, k_order=k_order)
print(f"an shape: {an.shape}")
# plot
for i in range(0, len(interval) - 1):
    x, y = get_plot_point(an[i], interval[i], interval[i + 1], k_order=k_order, num=5)
    plt.plot(x, y, color='black')


plt.plot(x, y, label='DG solution', color='black')
plt.legend(loc="upper right")
plt.show()