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

    lpos='cm',
    lopt='above, xshift=3mm, yshift=2mm',
    lbox=True,
    )

plot.height = plot.width

plot.axes()

CDW = u > 1e-3
SYM = ~CDW

plot.line(doping[CDW], u[CDW] * 1e2, color='cyan', mark='triangle*')
plot.line(doping[SYM], u[SYM] * 1e2, color='cyan', mark='triangle*',
    label=r'$|u|_{2 \times 2}$ (pm)')

plot.line(doping[CDW], wlog[CDW] * 1e3, color='gray', mark='x')
plot.line(doping[SYM], wlog[SYM] * 1e3, color='gray', mark='x',
    label=r'$\omega_{\mathrm{log}}$ (meV)')

plot.line(doping[CDW], w2nd[CDW] * 1e3, color='lightgray', mark='+')
plot.line(doping[SYM], w2nd[SYM] * 1e3, color='lightgray', mark='+',
    label=r'$\omega_{\mathrm{2nd}}$ (meV)')

plot.line(doping[CDW], lamda[CDW], color='red', mark='square*')
plot.line(doping[SYM], lamda[SYM], color='red', mark='square*',
    label=r'$\lambda$')

plot.line(doping[CDW], Tc[CDW], color='blue', mark='*')
plot.line(doping[SYM], Tc[SYM], color='blue', mark='*',
    label=r'$T_{\mathrm c}$ (K)')

doping, lamda, wlog, w2nd, Tc, u = np.loadtxt('relax_large.dat', skiprows=1).T

style = dict(only_marks=True, mark_size='0.75pt')

plot.line(doping, u * 1e2, mark='triangle*', **style)
plot.line(doping, wlog * 1e3, mark='x', **style)
plot.line(doping, w2nd * 1e3, mark='+', **style)
plot.line(doping, lamda, mark='square*', **style)
plot.line(doping, Tc, mark='*', **style)

plot.save('phases.pdf')
