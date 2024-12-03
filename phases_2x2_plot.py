#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np

colors = ['black', 'teal', 'gray', 'lightgray', 'orange', 'tomato', 'chocolate']

def phases(wmin, u, u_thr=2e-3):
    SYM = (wmin >= 0) & (abs(u) <= u_thr)
    CDW = (wmin >= 0) & (u > u_thr)
    return SYM, CDW

xel, dE, N0, mu, u, lamda, wlog, w2nd, wmin, Tc = np.loadtxt('phases_2x2.dat',
    skiprows=1).T

for label, phase in enumerate(phases(wmin, u)):
    plt.plot(xel[phase], Tc[phase], 'o', color=colors[0],
        label=r'$T_{\mathrm{c}}$ (K)' if label else None)

    plt.plot(xel[phase], lamda[phase], 's', color=colors[1],
        label=r'$\lambda$' if label else None)

    plt.plot(xel[phase], wlog[phase] * 1e3, 'x', color=colors[2],
        label=r'$\omega_{\mathrm{log}}$ (meV)' if label else None)

    plt.plot(xel[phase], w2nd[phase] * 1e3, '+', color=colors[3],
        label=r'$\overline{\omega}_2$ (meV)' if label else None)

    plt.plot(xel[phase], N0[phase] * elphmod.misc.Ry, 'D', color=colors[4],
        label=r'$N(\varepsilon_F)$ (1/Ry)' if label else None)

    plt.plot(xel[phase], -dE[phase] * 1e3, '*', color=colors[5],
        label=r'$\Delta E$ (meV)' if label else None)

    plt.plot(xel[phase], u[phase] * 1e2, 'v', color=colors[6],
        label=r'$|u|$ (pm)' if label else None)

nel, xel, dE, N0, mu, u, lamda, wlog, w2nd, wmin, Tc = np.loadtxt(
    'phases_18sqrt3/from_2x2.dat', skiprows=1).T

for phase in phases(wmin, u):
    plt.plot(xel[phase], Tc[phase], color=colors[0])
    plt.plot(xel[phase], lamda[phase], color=colors[1])
    plt.plot(xel[phase], wlog[phase] * 1e3, color=colors[2])
    plt.plot(xel[phase], w2nd[phase] * 1e3, color=colors[3])
    plt.plot(xel[phase], N0[phase] * elphmod.misc.Ry, color=colors[4])
    plt.plot(xel[phase], -dE[phase] * 1e3, color=colors[5])
    plt.plot(xel[phase], u[phase] * 1e2, color=colors[6])

plt.title('Lines are for reference from Supplemental Figure 7')
plt.xlabel('Doping electrons per unit cell')
plt.legend()
plt.show()
