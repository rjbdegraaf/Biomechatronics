import numpy as np

# Arm segment lengths
L1 = 1  # Shoulder to elbow
L2 = 1.5  # Elbow to wrist
L3 = 1  # Wrist to hand

# Generate circular path from rest to cup
radius = 3
center_x = radius
center_y = -(L1 + L2 + L3)  # Center of the circle below the arm

angles = np.linspace(np.pi, 0.5 * np.pi, 100)
target_xs = radius * np.cos(angles) + center_x
target_ys = radius * np.sin(angles) + center_y

# Generate second circular path from the end of the first path
second_radius = 2
second_center_x = target_xs[-1]  # End point of the first path
second_center_y = target_ys[-1] + second_radius  # Center below the end point

second_angles = np.linspace(1.5 * np.pi, 1 * np.pi, 100)
second_target_xs = second_radius * np.cos(second_angles) + second_center_x
second_target_ys = second_radius * np.sin(second_angles) + second_center_y

# Combine both paths
target_xs = np.concatenate((target_xs, second_target_xs))
target_ys = np.concatenate((target_ys, second_target_ys))