from math import cos, pi, sin

import numpy as np
import sympy as sp


def run_part_1():
    # Params
    theta = -pi / 2.0
    alpha = pi / 6.0
    gamma = pi / 4.0
    l1 = 1.0
    l2 = 1.0
    lb = 0.5
    x = 0.8
    y = 0.8

    # We want to find Ges which is from e to s
    # Need to find Ge2, then G21, then G1b, then Gbs

    Ge2 = np.array([[0, 1, l2], [-1, 0, 0], [0, 0, 1]])

    G21 = np.array(
        [[cos(gamma), -sin(gamma), l1], [sin(gamma), cos(gamma), 0], [0, 0, 1]]
    )

    G1b = np.array(
        [[cos(alpha), -sin(alpha), lb], [sin(alpha), cos(alpha), 0], [0, 0, 1]]
    )

    Gbs = np.array(
        [[cos(theta), -sin(theta), x], [sin(theta), cos(theta), y], [0, 0, 1]]
    )

    Ges = Gbs @ G1b @ G21 @ Ge2

    # Symbolic versions of the same transforms
    theta_sym, alpha_sym, gamma_sym = sp.symbols("theta alpha gamma", real=True)
    l1_sym, l2_sym, lb_sym, x_sym, y_sym = sp.symbols("l1 l2 lb x y", real=True)

    Ge2_sym = sp.Matrix([[0, 1, l2_sym], [-1, 0, 0], [0, 0, 1]])

    G21_sym = sp.Matrix(
        [
            [sp.cos(gamma_sym), -sp.sin(gamma_sym), l1_sym],
            [sp.sin(gamma_sym), sp.cos(gamma_sym), 0],
            [0, 0, 1],
        ]
    )

    G1b_sym = sp.Matrix(
        [
            [sp.cos(alpha_sym), -sp.sin(alpha_sym), lb_sym],
            [sp.sin(alpha_sym), sp.cos(alpha_sym), 0],
            [0, 0, 1],
        ]
    )

    Gbs_sym = sp.Matrix(
        [
            [sp.cos(theta_sym), -sp.sin(theta_sym), x_sym],
            [sp.sin(theta_sym), sp.cos(theta_sym), y_sym],
            [0, 0, 1],
        ]
    )

    Ges_sym = Gbs_sym * G1b_sym * G21_sym * Ge2_sym
    Ges_sym = sp.simplify(Ges_sym)
    sp.pprint(Ges_sym)

    p1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p2 = np.array([0.5, 1, 1], dtype=np.float64)

    p1s = Ges @ p1
    p2s = Ges @ p2
    print("P1 in s is " + str(p1s))
    print("P2 in s is " + str(p2s))

    Geb = G1b @ G21 @ Ge2
    qB = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    qE = np.linalg.inv(Geb) @ qB
    print("Point q in E is " + str(qE))
