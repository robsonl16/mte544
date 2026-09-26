# Robot Params
import time
from math import cos, pi, sin

import matplotlib.pyplot as plt
import numpy as np

WHEEL_RADIUS = 0.05  # meters
TRACK_WIDTH = 0.3  # meters

TIME_STEP = 0.1  # seconds


def compute_pose(v, omega, dt, x, y, theta):
    theta_new = (theta + omega * dt) % (2 * pi)
    x_new = x + v * dt * cos(theta)
    y_new = y + v * dt * sin(theta)
    return x_new, y_new, theta_new


def plot_pose_history(pose_history, dt):
    """Plot the robot's path and heading over time.

    pose_history should contain rows of (x, y, theta), including the initial
    pose. Angles are in radians and dt is in seconds.
    """
    poses = np.asarray(pose_history, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) == 0:
        raise ValueError(
            "pose_history must be a non-empty sequence of (x, y, theta) rows"
        )

    times = np.arange(len(poses)) * dt
    x, y, theta = poses.T

    fig, (path_ax, heading_ax) = plt.subplots(1, 2, figsize=(12, 5))

    path_ax.plot(x, y, color="tab:blue", label="Path")
    path_ax.scatter(x[0], y[0], color="tab:green", zorder=3, label="Start")
    path_ax.scatter(x[-1], y[-1], color="tab:red", zorder=3, label="End")

    # Draw a few arrows so the robot's orientation is visible along its path.
    arrow_stride = max(1, len(poses) // 12)
    arrow_indices = np.arange(0, len(poses), arrow_stride)
    arrow_length = 0.08 * max(np.ptp(x), np.ptp(y), 1.0)
    path_ax.quiver(
        x[arrow_indices],
        y[arrow_indices],
        np.cos(theta[arrow_indices]) * arrow_length,
        np.sin(theta[arrow_indices]) * arrow_length,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="tab:orange",
        label="Heading",
    )
    path_ax.set_title("Robot path")
    path_ax.set_xlabel("x (m)")
    path_ax.set_ylabel("y (m)")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(True, alpha=0.3)
    path_ax.legend()

    # Unwrap for display so crossing 2π does not look like a sudden jump.
    heading_ax.plot(times, np.unwrap(theta), color="tab:purple")
    heading_ax.set_title("Heading over time")
    heading_ax.set_xlabel("Time (s)")
    heading_ax.set_ylabel("θ (rad, unwrapped)")
    heading_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, (path_ax, heading_ax)


def main():
    x = 0.0  # initial x position in meters
    y = 0.0  # initial y position in meters
    theta = 0.0  # initial orientation in radians
    TOTAL_TIME = 30.0  # seconds

    pose_history = [(x, y, theta)]
    sim_time = 0.0  # seconds

    while sim_time < TOTAL_TIME:
        v = 1.0 + 0.2 * sin(sim_time)
        omega = 0.2 + 0.6 * cos(sim_time)
        x, y, theta = compute_pose(v, omega, TIME_STEP, x, y, theta)
        pose_history.append((x, y, theta))
        print(f"Time: {sim_time:.2f}s, Pose: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
        time.sleep(TIME_STEP)
        sim_time += TIME_STEP

    plot_pose_history(pose_history, TIME_STEP)
    plt.show()


if __name__ == "__main__":
    main()
