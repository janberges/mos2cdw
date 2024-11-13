#!/usr/bin/env python3

import elphmod
import numpy as np
import os
import storylines

nel1, xel1, dE1, mu1, u1, lamda1, wlog1, w2nd1, wmin1, Tc1 = np.loadtxt(
    'polaron_new.dat', skiprows=1).T

nel2, xel2, dE2, mu2, u2, lamda2, wlog2, w2nd2, wmin2, Tc2 = np.loadtxt(
    'cdw_new.dat', skiprows=1).T

u_thr = 2e-3

pol = (wmin1 >= 0) & (u1 >= u_thr)
sym = (wmin1 >= 0) & (abs(u1) < u_thr)
cdw = (wmin2 >= 0) & (u2 >= u_thr)

(nels, xels, dEs, mus, us, lamdas, wlogs, w2nds, wmins, Tcs) = (nel1[sym],
    xel1[sym], dE1[sym], mu1[sym], u1[sym], lamda1[sym], wlog1[sym], w2nd1[sym],
    wmin1[sym], Tc1[sym])

(nelp, xelp, dEp, mup, up, lamdap, wlogp, w2ndp, wminp, Tcp) = (nel1[pol],
    xel1[pol], dE1[pol], mu1[pol], u1[pol], lamda1[pol], wlog1[pol], w2nd1[pol],
    wmin1[pol], Tc1[pol])

(nelc, xelc, dEc, muc, uc, lamdac, wlogc, w2ndc, wminc, Tcc) = (nel2[cdw],
    xel2[cdw], dE2[cdw], mu2[cdw], u2[cdw], lamda2[cdw], wlog2[cdw], w2nd2[cdw],
    wmin2[cdw], Tc2[cdw])

plot = storylines.Plot(
    style='APS',
    font='Utopia',

    width=9.0,
    height=7.0,

    xlabel='Doping electrons per unit cell',
    ylabel='Critical temperature (K)',

    xstep=0.1,
    ystep=5.0,
    ymax=22.5,

    grid=True,

    lpos='cb',
    lopt='above=5mm',
    lbox=True,

    mark_size='0.8pt',
    )

plot.line(xels, Tcs, color='blue', thick=True, label='undistorted')

plot.line(xelc, Tcc, color='orange', thick=True, label=r'$2 \times 2$ CDW')

for group in elphmod.misc.group(nelp, 1.1):
    plot.line(xelp[group], Tcp[group], color='brown', thick=True,
        mark='*' if len(group) == 1 else None,
        label='other' if len(group) > 1 else None)

plot.save('tc_18sqrt3.pdf')

plot.width = 5.0
plot.height = 2.5
plot.lput = False

plot.xlabel = None
plot.ylabel = None

plot.xclose = True
plot.yclose = True

plot.yticks = [0.0, 10.0, (20.0, None)]

plot.left = 0.4
plot.bottom = 0.4

plot.node(0, 19, r'$T_{\mathrm c}$ (K)', rotate=90, above=True)
plot.node(0.68, 0, r'$x$', below='1mm')

os.makedirs('here', exist_ok=True)

for i in range(len(xel1)):
    plot.line(x=xel1[i], zindex=0)

    plot.save('here/%03d.pdf' % nel1[i])

    plot.lines.pop(0)
