import numpy as np

"""
a = 0.8          # Muscle activation (0 to 1)
l = 1.1          # Normalized muscle length (1.0 = optimal length)
v = -0.2         # Normalized contraction velocity (<0: shortening, >0: lengthening)

l0 = 1.0         # Optimal muscle length
vmax = 1.0       # Maximum contraction velocity (normalized)
F_max = 1000.0   # Maximum isometric force
k_passive = 50.0 # Passive stiffness coefficient
"""

def muscle_force(a, l, v, l0=1.0, vmax=1.0, F_max=1000.0, k_passive=50.0):
    # --- Force–length relationship (bell-shaped curve) ---
    f_l = np.exp(-((l - l0)/0.3)**2)

    # --- Force–velocity relationship ---
    if v < 0:
        f_v = (vmax + v) / (vmax - 0.25 * v)
    else:
        f_v = 1.8 - 0.8 * v

    f_v = max(0.0, f_v)  # Clamp to prevent negative force

    # --- Passive force ---
    f_passive = k_passive * max(0, l - l0)

    # --- Total force ---
    return a * f_l * f_v * F_max + f_passive

