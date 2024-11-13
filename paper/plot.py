#!/usr/bin/env python3

import elphmod
import numpy as np
import matplotlib.pyplot

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

matplotlib.rc('font', size=20)

fig, ax = matplotlib.pyplot.subplots(1, figsize=(7, 7))

fig.subplots_adjust(0.15, 0.15, 0.95, 0.95)

colors = sum([2 * [c] for c in ['azure', 'lightgray', 'mistyrose']], start=[])

nodes = np.array([0.0, 2.7, 3.0, 3.6, 4.0, 7.8])

x = np.linspace(nodes.min(), nodes.max(), 200)
ymin = 1.0
ymax = 100

normalize = matplotlib.colors.Normalize(vmin=x[0], vmax=x[-1])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',
    list(zip(normalize(nodes), colors)))

for i in range(x.size - 1):
    ax.fill_between([x[i], x[i + 1]], ymin, ymax, color=cmap(normalize(x[i])))

exp = np.loadtxt('exp_dome_1')
ax.scatter(exp[:, 0], exp[:, 1], fc='none', ec='black', s=100, label='[ref]')

pwi = elphmod.bravais.read_pwi('../dft/MoS2.pwi')
a = elphmod.bravais.primitives(**pwi) * 1e-8
vuc = np.linalg.norm(np.cross(a[0], a[1]))
scale = 1 / (1e14 * vuc)

ax.plot(xels * scale, Tcs, color='teal', label=r'$1 \times 1$ H')
ax.plot(xelc * scale, Tcc, color='coral', label=r'$2 \times 2$ CDW')

scatter = []
labeled = False

for group in elphmod.misc.group(nelp, 1.1):
    if len(group) == 1:
        scatter.extend(group)
    else:
        ax.plot(xelp[group] * scale, Tcp[group], color='slategray',
            label=None if labeled else 'other')

        labeled = True

ax.scatter(xelp[scatter] * scale, Tcp[scatter], c='slategray', s=10)

ax.set_xlabel('$n$ [$10^{14}\,\mathrm{cm}^{-2}$]')
ax.set_ylabel('$T_c$ [K]')

ax.tick_params('both', size=6, direction='in', top=True, right=True, which='major')
ax.tick_params('both', size=4, direction='in', top=True, right=True, which='minor')

ax.set_xlim(x[0], x[-1])
ax.set_ylim(ymin, ymax)

ax.set_yscale('log')

ax.legend(frameon=False, loc='upper left', handlelength=0.7, ncol=1, fontsize=16)

fig.savefig('plot.pdf')
