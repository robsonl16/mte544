import time
from math import cos, pi, sin

import matplotlib.pyplot as plt
import numpy as np


def compute_twist(u1, u2, u3, x, y, theta, dt):
    l = 0.25
    r = 0.1
    b1 = pi / 2
    b2 = -5 * pi / 6
    b3 = -pi / 6
    # G1 = np.array([[cos(theta + b1), sin(theta + b1), l]])
    # G2 = np.array([[cos(theta + b2), sin(theta + b2), -l]])
    # G3 = np.array([[cos(theta + b3), sin(theta + b3), l / 2]])
    G = np.array(
        [
            [cos(theta + b1), sin(theta + b1), l],
            [cos(theta + b2), sin(theta + b2), l],
            [cos(theta + b3), sin(theta + b3), l],
        ]
    )
    G[np.isclose(G, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    # G = np.array([G1, G2, G3])
    u = np.array([u1, u2, u3])
    vel_pose = np.linalg.solve(G, r * u)
    vel_pose[np.isclose(vel_pose, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    x_dot, y_dot, omega = vel_pose
    theta_new = (theta + omega * dt) % (2 * pi)
    x_new = x + x_dot * dt
    y_new = y + y_dot * dt
    return x_new, y_new, theta_new


def compute_twist_2(v, omega, x, y, theta, dt):
    l = 0.25
    r = 0.1
    b1 = pi / 2
    b2 = -5 * pi / 6
    b3 = -pi / 6
    # G1 = np.array([[cos(theta + b1), sin(theta + b1), l]])
    # G2 = np.array([[cos(theta + b2), sin(theta + b2), -l]])
    # G3 = np.array([[cos(theta + b3), sin(theta + b3), l / 2]])
    G = np.array(
        [
            [cos(theta + b1), sin(theta + b1), l],
            [cos(theta + b2), sin(theta + b2), l],
            [cos(theta + b3), sin(theta + b3), l],
        ]
    )
    G[np.isclose(G, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    # G = np.array([G1, G2, G3])
    vel_pose = np.array([v * cos(theta), v * sin(theta), omega])
    u = (1 / r) * G @ vel_pose
    u[np.isclose(u, 0.0, atol=1e-12, rtol=0.0)] = 0.0
    x_dot, y_dot, omega = vel_pose
    theta_new = (theta + omega * dt) % (2 * pi)
    x_new = x + x_dot * dt
    y_new = y + y_dot * dt
    return x_new, y_new, theta_new, u


def plot_pose_history(pose_history, dt, case_name="Robot pose"):
    """Plot x vs y and each pose variable against time for one case.

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    path_ax, x_ax, y_ax, theta_ax = axes.flat

    path_ax.plot(x, y, color="tab:blue", label="Path")
    path_ax.scatter(x[0], y[0], color="tab:green", zorder=3, label="Start")
    path_ax.scatter(x[-1], y[-1], color="tab:red", zorder=3, label="End")
    path_ax.set_title("Top view: y vs x")
    path_ax.set_xlabel("x (m)")
    path_ax.set_ylabel("y (m)")
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.grid(True, alpha=0.3)
    path_ax.legend()

    for axis, values, name, color, units in (
        (x_ax, x, "x position", "tab:blue", "m"),
        (y_ax, y, "y position", "tab:green", "m"),
        (theta_ax, theta, "orientation θ", "tab:purple", "rad"),
    ):
        axis.plot(times, values, color=color)
        axis.set_title(f"{name} vs time")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"{name} ({units})")
        axis.grid(True, alpha=0.3)

    fig.suptitle(case_name)
    fig.tight_layout()
    return fig, axes


def run_part_3():
    x = 0.0  # initial x position in meters
    y = 0.0  # initial y position in meters
    theta = 0  # initial orientation in radians
    TIME_STEP = 0.1  # seconds
    TOTAL_TIME = 30.0  # seconds
    u_history = []

    pose_history = [(x, y, theta)]
    sim_time = 0.0  # seconds

    while sim_time < TOTAL_TIME:
        # u1 = -2
        # u2 = 1
        # u3 = 1
        v = 1
        omega = 1
        # x, y, theta = compute_twist(u1, u2, u3, x, y, theta, TIME_STEP)
        x, y, theta, u = compute_twist_2(v, omega, x, y, theta, TIME_STEP)
        pose_history.append((x, y, theta))
        u_history.append(u)
        print(f"Time: {sim_time:.2f}s, Pose: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
        # time.sleep(TIME_STEP)
        sim_time += TIME_STEP

    plot_pose_history(pose_history, TIME_STEP)
    plt.show()
