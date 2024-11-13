#!/usr/bin/env python3

import elphmod
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import storylines

matplotlib.rc('font', size=16)

fig, axes = plt.subplots(2, 3, figsize=(21, 12))

fig.subplots_adjust(0.05, 0.05, 0.98, 0.98, wspace=0.05, hspace=0.05)

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

colors = sum([2 * [c] for c in ['azure', 'lightgray', 'mistyrose']], start=[])

nodes = np.array([0.0, 2.7, 3.0, 3.6, 4.0, 7.8])

x = np.linspace(nodes.min(), nodes.max(), 200)
ymin = 1.0
ymax = 100

normalize = matplotlib.colors.Normalize(vmin=x[0], vmax=x[-1])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',
    list(zip(normalize(nodes), colors)))

ax = axes[1, 0]

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
lines = []

for group in elphmod.misc.group(nelp, 1.1):
    if len(group) == 1:
        scatter.extend(group)
    else:
        lines.append(group)

for n, line in enumerate(lines):
    ax.plot(xelp[line] * scale, Tcp[line], color='slategray',
        label=None if n else 'other')

ax.scatter(xelp[scatter] * scale, Tcp[scatter], c='slategray', s=10)

ax.set_xlabel('$n$ [$10^{14}\,\mathrm{cm}^{-2}$]')
ax.set_ylabel('$T_c$ [K]')

ax.tick_params('both', size=6, direction='in', top=True, right=True, which='major')
ax.tick_params('both', size=4, direction='in', top=True, right=True, which='minor')

ax.set_xlim(x[0], x[-1])
ax.set_ylim(ymin, ymax)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

ax.set_yscale('log')

legendstyle = dict(frameon=False, handlelength=0.7, ncol=1)

ax.legend(loc='upper left', fontsize=16, **legendstyle)

def load_xyz(xyz):
    with open(xyz) as lines:
        nat = int(next(lines))
        a = np.reshape(list(map(float, next(lines).split()[-9:])), (3, 3)).T

        r = np.empty((nat, 3))
        typ = []

        for i in range(nat):
            col = next(lines).split()

            typ.append(col[0])
            r[i] = list(map(float, col[-3:]))

    return a, typ, r

cmap = plt.get_cmap('Greys_r')

for ax, label in [
        (axes[0, 0], 'polaron248'),
        (axes[0, 1], 'polaron259'),
        (axes[0, 2], 'polaron333'),
        (axes[1, 1], 'cdw462'),
        (axes[1, 2], 'polaron462'),
        ]:

    A, typ, R0 = load_xyz('symmetric.xyz')

    A, typ, R = load_xyz('xyz/%s.xyz' % label)

    u = R - R0

    tau = np.linalg.norm(R0[0, :2] - R0[1, :2])

    sgn = -1 if np.allclose(A[0, 1:], 0.0) else +1

    for na in range(len(R)):
        if np.isclose(R0[na] @ A[0],
                np.linalg.norm(R0[na]) * np.linalg.norm(A[0])):

            R0[na] += A[1]

        if R0[na] @ A[0] < sgn * 1e-5:
            R0[na] += A[0]

        if R0[na] @ A[0] > 0.75 * np.linalg.norm(A[0]) ** 2 - sgn * 1e-5:
            R0[na] -= A[0]

        if sgn > 0:
            phi = -np.arctan2(A[0, 1], A[0, 0])

            u[na] = elphmod.bravais.rotate(u[na], phi, two_dimensional=False)
            R0[na] = elphmod.bravais.rotate(R0[na], phi, two_dimensional=False)

    R = R0 + u

    bonds = np.array(storylines.bonds(R1=R[0::3, :2], R2=R[1::3, :2],
        dmin=0.1 * tau, dmax=1.5 * tau))

    lengths = np.linalg.norm(bonds[:, 1] - bonds[:, 0], axis=1)

    normalize = matplotlib.colors.Normalize(lengths.min(), lengths.max())

    for bond, length in zip(bonds, lengths):
        ax.plot(*zip(*bond), linewidth=3, color=cmap(normalize(length)))

    atom = dict(mark='*', only_marks=True)

    ax.scatter(R[1::3, 0], R[1::3, 1], s=20, zorder=2, color='#5f97e6')
    ax.scatter(R[0::3, 0], R[0::3, 1], s=20, zorder=2, color='#d8ba8d')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('equal')
    ax.axis('off')

    pad = 2.0

    ax.set(
        xlim=(R[:, 0].min() - pad, R[:, 0].max() + pad),
        ylim=(R[:, 1].min() - pad, R[:, 1].max() + pad))

    scale = 20.0

    arrow = {'->': True}

    ok = np.linalg.norm(u[:, :2], axis=1) > 0.075

    ax.quiver(*R[ok, :2].T, *scale * u[ok, :2].T, angles='xy', scale_units='xy',
        scale=1, width=0.005, headwidth=4, headlength=3, headaxislength=3,
        zorder=3)

fig.savefig('plot.pdf')

fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey='row',
    width_ratios=(nels.ptp(), nelc.ptp(), nelp.ptp()))

fig.subplots_adjust(0.20, 0.12, 0.97, 0.94, wspace=0.05)

colors = ['black', 'teal', 'gray', 'lightgray', 'tomato', 'chocolate']

ax[0].plot(xels * scale, Tcs, color=colors[0], label=r'$T_{\mathrm{c}}$ [K]')
ax[0].plot(xels * scale, lamdas, color=colors[1], label=r'$\lambda$')
ax[0].plot(xels * scale, wlogs * 1e3, color=colors[2],
    label=r'$\omega_{\mathrm{log}}$ [meV]')
ax[0].plot(xels * scale, w2nds * 1e3, color=colors[3],
    label=r'$\omega_{\mathrm{2nd}}$ [meV]')
ax[0].plot(xels * scale, -dEs * 1e3, color=colors[4], label=r'$\Delta E$ [meV]')
ax[0].plot(xels * scale, us * 1e2, color=colors[5], label=r'$|u|$ [pm]')

ax[0].set_title(r'$1 \times 1$ H')
ax[0].set_xlim(xels.min() * scale, xels.max() * scale)
ax[0].legend(loc='right', bbox_to_anchor=(-0.15, 0.5), **legendstyle)

ax[1].plot(xelc * scale, Tcc, color=colors[0])
ax[1].plot(xelc * scale, lamdac, color=colors[1])
ax[1].plot(xelc * scale, wlogc * 1e3, color=colors[2])
ax[1].plot(xelc * scale, w2ndc * 1e3, color=colors[3])
ax[1].plot(xelc * scale, -dEc * 1e3, color=colors[4])
ax[1].plot(xelc * scale, uc * 1e2, color=colors[5])

ax[1].set_title(r'$2 \times 2$ CDW')
ax[1].set_xlim(xelc.min() * scale, xelc.max() * scale)
ax[1].set_xlabel('$n$ [$10^{14}\,\mathrm{cm}^{-2}$]')

for line in lines:
    ax[2].plot(xelp[line] * scale, Tcp[line], color=colors[0])
    ax[2].plot(xelp[line] * scale, lamdap[line], color=colors[1])
    ax[2].plot(xelp[line] * scale, wlogp[line] * 1e3, color=colors[2])
    ax[2].plot(xelp[line] * scale, w2ndp[line] * 1e3, color=colors[3])
    ax[2].plot(xelp[line] * scale, -dEp[line] * 1e3, color=colors[4])
    ax[2].plot(xelp[line] * scale, up[line] * 1e2, color=colors[5])

ax[2].scatter(xelp[scatter] * scale, Tcp[scatter], c=colors[0], s=10)
ax[2].scatter(xelp[scatter] * scale, lamdap[scatter], c=colors[1], s=10)
ax[2].scatter(xelp[scatter] * scale, wlogp[scatter] * 1e3, c=colors[2], s=10)
ax[2].scatter(xelp[scatter] * scale, w2ndp[scatter] * 1e3, c=colors[3], s=10)
ax[2].scatter(xelp[scatter] * scale, -dEp[scatter] * 1e3, c=colors[4], s=10)
ax[2].scatter(xelp[scatter] * scale, up[scatter] * 1e2, c=colors[5], s=10)

ax[2].set_title('other')
ax[2].set_xlim(xelp.min() * scale, xelp.max() * scale)

for a in ax:
    a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

fig.savefig('plot_si.pdf')
