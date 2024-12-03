#!/usr/bin/env python3

import numpy as np
import elphmod

def setup_coupling(
    t_z2=-0.14,
    t_z2_x2y2=0.48,
    t_z2_xy=-0.38,
    t_x2y2=-0.26,
    t_x2y2_xy=0.31,
    t_xy=0.32,

    a=3.185,

    M=95.95,
    m=32.06,

    beta=5.0):

    t0 = np.array([
        [ t_z2,      t_z2_x2y2, t_z2_xy  ],
        [ t_z2_x2y2, t_x2y2,    t_x2y2_xy],
        [-t_z2_xy,  -t_x2y2_xy, t_xy     ],
        ]) / elphmod.misc.Ry

    a = a / elphmod.misc.a0

    def R(phi):
        return np.array([
            [1,               0,                0],
            [0, np.cos(2 * phi), -np.sin(2 * phi)],
            [0, np.sin(2 * phi),  np.cos(2 * phi)],
            ])

    def dR_dphi(phi):
        return 2 * np.array([
            [0,                0,                0],
            [0, -np.sin(2 * phi), -np.cos(2 * phi)],
            [0,  np.cos(2 * phi), -np.sin(2 * phi)],
            ])

    def derivative(t0, phi):
        dt_dr = -beta / a * np.einsum('ab,bc,cd->ad', R(phi), t0, R(-phi))

        dt_dphi = (
              np.einsum('ab,bc,cd->ad', dR_dphi(phi), t0, R(-phi))
            - np.einsum('ab,bc,cd->ad', R(phi), t0, dR_dphi(-phi))
            )

        dt_dx = dt_dr * np.cos(phi) - dt_dphi / a * np.sin(phi)
        dt_dy = dt_dr * np.sin(phi) + dt_dphi / a * np.cos(phi)

        return np.array([dt_dx, dt_dy])

    dt = np.zeros((6, 9, 3, 3))

    for n in range(6):
        dt[n, 3:5] = derivative(t0.T if n % 2 else t0,
            n * 60 * elphmod.bravais.deg)

    sqrtM = np.sqrt(np.repeat([m, M, m], 3)[:, np.newaxis, np.newaxis]
        * elphmod.misc.uRy)

    def coupling(q1=0, q2=0, q3=0, k1=0, k2=0, k3=0, **ignore):
        K1 = k1 + q1
        K2 = k2 + q2

        return (
            + dt[0] * (np.exp(1j * K1) - np.exp(1j * k1))
            + dt[3] * (np.exp(-1j * K1) - np.exp(-1j * k1))
            + dt[1] * (np.exp(1j * (K1 + K2)) - np.exp(1j * (k1 + k2)))
            + dt[4] * (np.exp(-1j * (K1 + K2)) - np.exp(-1j * (k1 + k2)))
            + dt[2] * (np.exp(1j * K2) - np.exp(1j * k2))
            + dt[5] * (np.exp(-1j * K2) - np.exp(-1j * k2))
            ) / sqrtM

    return coupling

def save_coupling(stem, coupling, el, ph):
    nq = nk = (3, 3, 1)

    q = elphmod.bravais.mesh(*nq, flat=True)
    g = elphmod.elph.sample(coupling, q, nk)

    elph = elphmod.elph.Model(el=el, ph=ph)
    elphmod.elph.q2r(elph, nq, nk, g, r=np.repeat(ph.r[1:2], el.size, axis=0))
    elph.standardize(eps=1e-10)
    elph.to_epmatwp(stem)
