#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines

nel1, xel1, dE1, mu1, u1, lamda1, wlog1, w2nd1, Tc1 = np.loadtxt('polaron.dat',
    skiprows=1).T

nel2, xel2, dE2, mu2, u2, lamda2, wlog2, w2nd2, Tc2 = np.loadtxt('cdw.dat',
    skiprows=1).T

pol = u1 > 2e-3
sym = ~pol
cdw = u2 > 2e-3

(nel0, xel0, dE0, mu0, u0, lamda0, wlog0, w2nd0, Tc0) = (nel1[sym], xel1[sym],
    dE1[sym], mu1[sym], u1[sym], lamda1[sym], wlog1[sym], w2nd1[sym], Tc1[sym])

(nel1, xel1, dE1, mu1, u1, lamda1, wlog1, w2nd1, Tc1) = (nel1[pol], xel1[pol],
    dE1[pol], mu1[pol], u1[pol], lamda1[pol], wlog1[pol], w2nd1[pol], Tc1[pol])

(nel2, xel2, dE2, mu2, u2, lamda2, wlog2, w2nd2, Tc2) = (nel2[cdw], xel2[cdw],
    dE2[cdw], mu2[cdw], u2[cdw], lamda2[cdw], wlog2[cdw], w2nd2[cdw], Tc2[cdw])

plot = storylines.Plot(
    style='APS',
    font='Utopia',

    width=9.0,
    height=7.0,

    xlabel='Doping electrons per unit cell',
    ylabel='Critical temperature (K)',

    xstep=0.1,

    grid=True,

    lpos='cb',
    lopt='above=2mm',
    lbox=True)

plot.line(xel0, Tc0, color='blue', label='undistorted')

for group in elphmod.misc.group(nel1, 1.1):
    plot.line(xel1[group], Tc1[group], color='brown',
        mark='asterisk' if len(group) == 1 else None,
        label='polaron' if len(group) == 1 else None)

plot.line(xel2, Tc2, color='orange', label=r'$2 \times 2$ CDW')

plot.save('tc_18sqrt3.pdf')

plot.width = 5.0
plot.height = 2.5
plot.lput = False

plot.xlabel = '$x$'
plot.ylabel = r'$T_{\mathrm c}$ (K)'

plot.xclose = True
plot.yclose = True

plot.xticks = [0.0, (0.1, None), (0.2, None), (0.3, None), (0.4, None), (0.5, None), 0.6]
plot.yticks = [0.0, (10.0, None), (20.0, None), 30.0]

plot.left = 0.5
plot.bottom = 0.5

for label, color, nel, xel, Tc in [
    ('polaron', 'blue', nel0, xel0, Tc0),
    ('polaron', 'brown', nel1, xel1, Tc1),
    ('cdw', 'orange', nel2, xel2, Tc2),
    ]:

    for i in range(len(nel)):
        print('%s%03d' % (label, nel[i]))

        plot.line(xel[i], Tc[i], mark='*', color=color)

        plot.save('png/%s%03dtc.png' % (label, nel[i]))

        plot.lines.pop()
