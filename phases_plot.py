#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines

nel, xel, dE, mu, u, lamda, wlog, w2nd, wmin, Tc = np.loadtxt('phases.dat',
    skiprows=1).T

plot = storylines.Plot(
    style='APS',
    font='Utopia',

    xlabel=r'Doping electrons per unit cell',

    xstep=0.1,
    ystep=5.0,

    grid=True,

    lpos='cm',
    lopt='above, xshift=3mm, yshift=2mm',
    lbox=True,
    )

plot.height = plot.width

plot.axes()

CDW = (wmin >= 0) & (u > 1e-3)
SYM = (wmin >= 0) & (u <= 1e-3)

plot.line(xel[CDW], u[CDW] * 1e2, color='cyan', mark='triangle*')
plot.line(xel[SYM], u[SYM] * 1e2, color='cyan', mark='triangle*',
    label=r'$|u|_{2 \times 2}$ (pm)')

plot.line(xel[CDW], wlog[CDW] * 1e3, color='gray', mark='x')
plot.line(xel[SYM], wlog[SYM] * 1e3, color='gray', mark='x',
    label=r'$\omega_{\mathrm{log}}$ (meV)')

plot.line(xel[CDW], w2nd[CDW] * 1e3, color='lightgray', mark='+')
plot.line(xel[SYM], w2nd[SYM] * 1e3, color='lightgray', mark='+',
    label=r'$\omega_{\mathrm{2nd}}$ (meV)')

plot.line(xel[CDW], lamda[CDW], color='red', mark='square*')
plot.line(xel[SYM], lamda[SYM], color='red', mark='square*',
    label=r'$\lambda$')

plot.line(xel[CDW], Tc[CDW], color='blue', mark='*')
plot.line(xel[SYM], Tc[SYM], color='blue', mark='*',
    label=r'$T_{\mathrm c}$ (K)')

nel, xel, dE, mu, u, lamda, wlog, w2nd, wmin, Tc = np.loadtxt('cdw.dat',
    skiprows=1).T

for i in range(len(Tc)):
    Tc[i] = elphmod.eliashberg.Tc(lamda[i], wlog[i], 0.0, w2nd[i], correct=True)

ok = (wmin >= 0) & (u >= 0)

style = dict(only_marks=True, mark_size='0.75pt')

plot.line(xel[ok], u[ok] * 1e2, mark='triangle*', **style)
plot.line(xel[ok], wlog[ok] * 1e3, mark='x', **style)
plot.line(xel[ok], w2nd[ok] * 1e3, mark='+', **style)
plot.line(xel[ok], lamda[ok], mark='square*', **style)
plot.line(xel[ok], Tc[ok], mark='*', **style)

plot.save('phases.pdf')
