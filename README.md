# Biomechatronics

This project simulates a 3-joint 2D arm (shoulder, elbow, wrist) actuated by real and artificial muscles. The simulation visualizes joint movements, muscle activations (including pneumatic artificial muscles, or PAMs), and the effects of activation noise while tracking a desired trajectory.

## Features

- **Inverse Kinematics** to compute target joint angles from hand positions.
- **PD Control** with optional activation noise.
- **Muscle Model** for force computation with optional PAM compensation.
- **Real-time Animation** using `matplotlib` to visualize:
  - Arm motion and trajectory
  - Muscle activations (real + PAM)
  - Noise and PAM contribution per joint

## Visualizations

- **Arm Movement:** 3-segment arm reaching toward a path of target positions.
- **Muscle Lines:** 12 visualized muscles (positive/negative real + PAM for each joint), thickness and color indicate activation and contraction.
- **Bar Plot:** Real-time activation levels for each muscle.
- **Time Plots:** Historical plots of activation noise and PAM compensation per joint.

## Installation

1. Clone the repository and ensure the following Python packages are installed:

    ```bash
    pip install numpy matplotlib
    ```

2. Ensure the following files are present:
   - `Muscle_Model.py` – contains the `muscle_force()` function.
   - `constants.py` – defines constants like `L1`, `L2`, `L3`, `target_xs`, and `target_ys`.

## Usage

Run the main script:

```bash
python Biomechatronics2DArm.py
```

This will launch a real-time animation window with all the visualizations.

## Configuration

You can toggle key simulation features by modifying the following flags in the script:

```python
apply_activation_noise = True  # Add Gaussian noise to activations
activation_noise_std = np.array([0.4, 0.4, 0.18])  # Std dev for noise per joint
apply_pam = True  # Enable or disable PAM-assisted actuation
```

## File Structure

- `simulate_arm.py`: Main simulation script with animation.
- `Muscle_Model.py`: Contains muscle force model.
- `constants.py`: Arm segment lengths and target trajectory.

## Credits

This simulation was developed as part of a biomechanics/control systems project using biologically inspired actuation models.
