#!/usr/bin/env python3

import numpy as np
import storylines

doping, lamda, wlog, w2nd, Tc, u = np.loadtxt('phases.dat', skiprows=1).T

plot = storylines.Plot(
    style='APS',
    font='Utopia',

    xlabel=r'Doping electrons per unit cell',

    xstep=0.1,
    ystep=5.0,

    grid=True,

    lpos='llrrrbbttt',
    lbox=True,
    )

plot.height = plot.width

plot.axes()

CDW = u > 0
SYM = ~CDW

plot.line(doping[CDW], u[CDW] * 1e2, color='cyan', mark='triangle*', jump=1.0)
plot.line(doping[SYM], u[SYM] * 1e2, color='cyan', mark='triangle*',
    label=r'$|u|_{2 \times 2}$ (pm)')

plot.line(doping[CDW], wlog[CDW] * 1e3, color='gray', mark='x', jump=1.0)
plot.line(doping[SYM], wlog[SYM] * 1e3, color='gray', mark='x',
    label=r'$\omega_{\mathrm{log}}$ (meV)')

plot.line(doping[CDW], w2nd[CDW] * 1e3, color='lightgray', mark='+', jump=1.0)
plot.line(doping[SYM], w2nd[SYM] * 1e3, color='lightgray', mark='+',
    label=r'$\omega_{\mathrm{2nd}}$ (meV)')

plot.line(doping[CDW], lamda[CDW], color='red', mark='square*', jump=1.0)
plot.line(doping[SYM], lamda[SYM], color='red', mark='square*',
    label=r'$\lambda$')

plot.line(doping[CDW], Tc[CDW], color='blue', mark='*', jump=1.0)
plot.line(doping[SYM], Tc[SYM], color='blue', mark='*',
    label=r'$T_{\mathrm c}$ (K)')

plot.save('phases.pdf')
