import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from Muscle_Model import muscle_force
from constants import L1, L2, L3, target_xs, target_ys

# Dynamics constants
I = np.array([1.0, 0.6, 0.3])
r = 0.03
dt = 0.02

apply_activation_noise = True
activation_noise_std = np.array([0.4, 0.4, 0.18])
apply_pam = True

q = np.array([-np.pi / 2, 0, 0])
q_dot = np.zeros(3)

cmap = mcolors.LinearSegmentedColormap.from_list("muscle_cmap", ['red', 'orange', 'green'])

def inverse_kinematics(x, y):
    hand_x, hand_y = x, y
    phi = 0
    wrist_x = hand_x - L3 * np.cos(phi)
    wrist_y = hand_y - L3 * np.sin(phi)
    dx = wrist_x
    dy = wrist_y
    r_ = np.hypot(dx, dy)
    if r_ > (L1 + L2):
        raise ValueError("Target is out of reach")
    cos_elbow = (r_**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_elbow = np.clip(cos_elbow, -1, 1)
    q_elbow = np.arccos(cos_elbow)
    k1 = L1 + L2 * np.cos(q_elbow)
    k2 = L2 * np.sin(q_elbow)
    q_shoulder = np.arctan2(dy, dx) - np.arctan2(k2, k1)
    q_wrist = - (q_shoulder + q_elbow)
    return np.array([q_shoulder, q_elbow, q_wrist])

def pd_activation(q_desired, q, q_dot, Kp=50, Kd=5):
    error = q_desired - q
    a_pos = np.clip(Kp * error - Kd * q_dot, 0, 1)
    a_neg = np.clip(-Kp * error + Kd * q_dot, 0, 1)
    return a_pos, a_neg

# Store history for plotting
time_data = []
pam_activations = [[] for _ in range(3)]
noise_data = [[] for _ in range(3)]

# Setup layout
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 4)

ax_arm = fig.add_subplot(gs[0:2, 0:3])
ax_bar = fig.add_subplot(gs[2, 0:3])
ax_joint_noise = [fig.add_subplot(gs[i, 3]) for i in range(3)]

# Axes setup
ax_arm.set_xlim(-5, 5)
ax_arm.set_ylim(-5, 5)
ax_arm.set_aspect('equal')
ax_arm.plot(target_xs, target_ys, 'b-', lw=1)

cup = plt.Rectangle((target_xs[99] - 0.2, target_ys[99]), 0.4, 0.6, color='orange')
ax_arm.add_patch(cup)
ax_arm.add_patch(plt.Rectangle((-0.4, -3.5), 0.8, 4, edgecolor='black', facecolor='none', linewidth=2))
ax_arm.add_patch(plt.Circle((0, 1.2), 0.55, edgecolor='black', facecolor='none', linewidth=2))

line1, = ax_arm.plot([], [], 'r-', lw=2)
line2, = ax_arm.plot([], [], 'g-', lw=2)
line3, = ax_arm.plot([], [], 'b-', lw=2)
muscle_lines = [ax_arm.plot([], [], lw=4)[0] for _ in range(12)]

muscle_labels = ['Shoulder Pos', 'Shoulder Neg', 'PAM Pos', 'PAM Neg',
                 'Elbow Pos', 'Elbow Neg', 'PAM Pos', 'PAM Neg',
                 'Wrist Pos', 'Wrist Neg', 'PAM Pos', 'PAM Neg']

bar_positions = np.arange(12)
bars = ax_bar.bar(bar_positions, [0]*12, color='tab:blue')
ax_bar.set_ylim(0, 1.1)
ax_bar.set_xticks(bar_positions)
ax_bar.set_xticklabels(muscle_labels, rotation=45, ha='right')
ax_bar.set_ylabel('Activation')
ax_bar.set_title('Muscle Activations (Real + PAM)')

# Time plots
lines_noise, lines_pam = [], []
for i, ax in enumerate(ax_joint_noise):
    ax.set_title(f'Joint {i} Noise & PAM')
    ax.set_xlim(0, len(target_xs))
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Value')
    lines_noise.append(ax.plot([], [], 'r-', label='Noise')[0])
    lines_pam.append(ax.plot([], [], 'b--', label='PAM')[0])
    ax.legend()

def get_muscle_color(length):
    norm_length = np.clip((length - 0.9) / (1.1 - 0.9), 0, 1)
    return cmap(norm_length)

def update(frame):
    global q, q_dot
    x, y = target_xs[frame], target_ys[frame]
    try:
        q_desired = inverse_kinematics(x, y)
    except ValueError:
        return []

    tau = np.zeros(3)
    activations = []
    frame_noise = []
    frame_pam = []

    for i in range(3):
        a_pos, a_neg = pd_activation(q_desired[i], q[i], q_dot[i])
        if apply_activation_noise:
            noise_pos = np.random.normal(0, activation_noise_std[i])
            noise_neg = np.random.normal(0, activation_noise_std[i])
            a_pos_noisy = np.clip(a_pos + noise_pos, 0, 1)
            a_neg_noisy = np.clip(a_neg + noise_neg, 0, 1)
            frame_noise.append((a_pos_noisy - a_pos + a_neg_noisy - a_neg)/2)
        else:
            a_pos_noisy, a_neg_noisy = a_pos, a_neg
            frame_noise.append(0)

        if apply_pam:
            pam_a_pos = np.clip(0.5 * a_pos_noisy, 0, 1)
            pam_a_neg = np.clip(0.5 * a_neg_noisy, 0, 1)
            frame_pam.append((pam_a_pos + pam_a_neg) / 2)
        else:
            pam_a_pos = pam_a_neg = 0
            frame_pam.append(0)

        l_pos = 1.0 + r * q[i]
        l_neg = 1.0 - r * q[i]
        v_pos = r * q_dot[i]
        v_neg = -r * q_dot[i]

        F_pos = muscle_force(a_pos_noisy, l_pos, v_pos)
        F_neg = muscle_force(a_neg_noisy, l_neg, v_neg)
        F_pam_pos = muscle_force(pam_a_pos, l_pos, v_pos) if apply_pam else 0
        F_pam_neg = muscle_force(pam_a_neg, l_neg, v_neg) if apply_pam else 0

        tau[i] = r * ((F_pos + F_pam_pos) - (F_neg + F_pam_neg))
        activations.extend([(a_pos_noisy, l_pos), (a_neg_noisy, l_neg),
                            (pam_a_pos, l_pos), (pam_a_neg, l_neg)])

    q_ddot = tau / I
    q_dot += q_ddot * dt
    q += q_dot * dt

    # Kinematics
    s_x, s_y = 0, 0
    e_x = s_x + L1 * np.cos(q[0])
    e_y = s_y + L1 * np.sin(q[0])
    w_x = e_x + L2 * np.cos(q[0] + q[1])
    w_y = e_y + L2 * np.sin(q[0] + q[1])
    h_x = w_x + L3 * np.cos(q[0] + q[1] + q[2])
    h_y = w_y + L3 * np.sin(q[0] + q[1] + q[2])

    line1.set_data([s_x, e_x], [s_y, e_y])
    line2.set_data([e_x, w_x], [e_y, w_y])
    line3.set_data([w_x, h_x], [w_y, h_y])

    joints = [(s_x, s_y, e_x, e_y), (e_x, e_y, w_x, w_y), (w_x, w_y, h_x, h_y)]
    segment_fraction = [0.15, 0.35, 0.65, 0.85]

    for i, (a, l) in enumerate(activations):
        j = i // 4
        x0, y0, x1, y1 = joints[j]
        direction = np.array([x1 - x0, y1 - y0])
        ortho = np.array([-direction[1], direction[0]])
        ortho = ortho / np.linalg.norm(ortho) * 0.1
        idx_in_group = i % 4
        frac = segment_fraction[idx_in_group]
        base_point = np.array([x0, y0]) + frac * direction
        offset_sign = 1 if idx_in_group < 2 else -1
        offset = ortho * offset_sign
        p0 = base_point + offset
        p1 = p0 + 0.05 * direction
        muscle_lines[i].set_data([p0[0], p1[0]], [p0[1], p1[1]])
        if idx_in_group < 2:
            muscle_lines[i].set_color(get_muscle_color(l))
            muscle_lines[i].set_linewidth(2 + 4 * a)
        else:
            muscle_lines[i].set_color((0, 0, 0.8, 0.7 + 0.3 * a))
            muscle_lines[i].set_linewidth(2 + 4 * a)

    for i, (a, _) in enumerate(activations):
        bars[i].set_height(a)
        bars[i].set_color('orangered' if i % 4 < 2 else 'dodgerblue')

    # Update history
    time_data.append(frame)
    for i in range(3):
        noise_data[i].append(frame_noise[i])
        pam_activations[i].append(frame_pam[i])
        lines_noise[i].set_data(time_data, noise_data[i])
        lines_pam[i].set_data(time_data, pam_activations[i])
        ax_joint_noise[i].set_xlim(0, max(100, frame))

    return [line1, line2, line3, *muscle_lines, *bars, *lines_noise, *lines_pam]

ani = FuncAnimation(fig, update, frames=len(target_xs), interval=20, blit=True)
plt.tight_layout()
plt.show()
