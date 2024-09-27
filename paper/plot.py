#!/usr/bin/env python3

import numpy as np
import storylines
import matplotlib.pyplot as plt

plot = storylines.Plot(
    style='APS',
    font='Utopia',

    xlabel=r'Doping electrons per unit cell',

    xstep=0.1,
    ystep=5.0,

    grid=True,

    lpos='tr',
    lopt='below right=1mm',
    lbox=True,
    )

plot.height = plot.width

for file in 'polaron', 'cdw':
    nel, xel, dE, mu, u, lamda, wlog, w2nd, Tc = np.loadtxt('%s.dat' % file,
        skiprows=1).T

    if file == 'polaron':
        mu0 = mu[0]

    for fun, x in (plot.line, xel), (plt.plot, nel):
        fun(x, -dE * 1e3, color='orange', label=r'$-\Delta E$ (meV)')
        fun(x, 10 * (mu - mu0), color='yellow', label=r'$10 \mu$ (eV)')
        fun(x, u * 1e2, color='cyan', label=r'$|u|_{2 \times 2}$ (pm)')
        fun(x, lamda, color='red', label=r'$\lambda$')
        fun(x, wlog * 1e3, color='gray', label=r'$\omega_{\mathrm{log}}$ (meV)')
        fun(x, w2nd * 1e3, color='lightgray', label=r'$\omega_{\mathrm{2nd}}$ (meV)')
        fun(x, Tc, color='blue', label=r'$T_{\mathrm{c}}$ (K)')

plt.legend()
plt.show()

plot.save('plot.pdf')
