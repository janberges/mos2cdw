#!/usr/bin/env python3

import elphmod
import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
import storylines

matplotlib.rc('font', size=20)

fig, axes = plt.subplots(2, 3, figsize=(21, 12))

fig.subplots_adjust(0.06, 0.06, 1.0, 0.98, wspace=0.1, hspace=0.05)

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

x = np.linspace(nodes.min(), nodes.max(), 1000)
ymin = 1.0
ymax = 100

normalize = matplotlib.colors.Normalize(vmin=x[0], vmax=x[-1])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',
    list(zip(normalize(nodes), colors)))

ax = axes[1, 0]

for i in range(x.size - 1):
    ax.fill_between([x[i], x[i + 1]], ymin, ymax, color=cmap(normalize(x[i])))

exp = np.loadtxt('exp_dome_1')
ax.scatter(exp[:, 0], exp[:, 1], fc='none', ec='black', s=100, label='ref.')

pwi = elphmod.bravais.read_pwi('../dft/MoS2.pwi')
a = elphmod.bravais.primitives(**pwi)
vuc = np.linalg.norm(np.cross(a[0], a[1])) * 1e-16
scale = 1 / (1e14 * vuc)

linestyle = dict(solid_capstyle='round', linewidth=2.5)

ax.plot(xels * scale, Tcs, color='teal', label=r'$1 \times 1$', **linestyle)
ax.plot(xelc * scale, Tcc, color='coral', label=r'$2 \times 2$', **linestyle)

scatter = []
lines = []

for group in elphmod.misc.group(nelp, 1.1):
    if len(group) == 1:
        scatter.extend(group)
    else:
        lines.append(group)

for n, line in enumerate(lines):
    ax.plot(xelp[line] * scale, Tcp[line], color='slategray',
        label=None if n else 'other', **linestyle)

ax.scatter(xelp[scatter] * scale, Tcp[scatter], c='slategray', s=20)

ax.set_xlabel('$n$ [$10^{14}\,\mathrm{cm}^{-2}$]')
ax.set_ylabel('$T_c$ [K]')

tickstyle = dict(direction='in', top=True, right=True)

ax.tick_params('both', size=6, which='major', **tickstyle)
ax.tick_params('both', size=4, which='minor', **tickstyle)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(ymin, ymax)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))

ax.set_yscale('log')

legendstyle = dict(frameon=False, handlelength=0.7, ncol=1)

ax.legend(loc='upper left', **legendstyle)

ax.text(-0.18, 0.95, '(d)', transform=ax.transAxes)

annotation = dict(textcoords='offset points', arrowprops=dict(arrowstyle='->'),
    ha='center', va='center')

n = np.argmax(nelp == 248)
ax.annotate('(a)', (xelp[n] * scale, Tcp[n]), (10, 40), **annotation)

n = np.argmax(nelp == 259)
ax.annotate('(b)', (xelp[n] * scale, Tcp[n]), (-10, -40), **annotation)

n = np.argmax(nelp == 333)
ax.annotate('(c)', (xelp[n] * scale, Tcp[n]), (-10, -40), **annotation)

n = np.argmax(nelc == 462)
ax.annotate('(e)', (xelc[n] * scale, Tcp[n]), (10, 40), **annotation)

n = np.argmax(nelp == 462)
ax.annotate('(f)', (xelp[n] * scale, Tcp[n]), (-10, -40), **annotation)

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

    return a * elphmod.misc.a0, typ, r * elphmod.misc.a0

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('grayscale',
    list(zip([0.0, 1.0], ['dimgray', 'white'])))

logS_min = +np.inf
logS_max = -np.inf

length_min = +np.inf
length_max = -np.inf

for abc, ax, label in [
        ('a', axes[0, 0], 'polaron248'),
        ('b', axes[0, 1], 'polaron259'),
        ('c', axes[0, 2], 'polaron333'),
        ('e', axes[1, 1], 'cdw462'),
        ('f', axes[1, 2], 'polaron462'),
        ]:

    A, typ, R0 = load_xyz('symmetric.xyz')

    A, typ, R = load_xyz('%s.xyz' % label)

    phi = -np.arctan2(A[0, 1], A[0, 0])

    b = np.array(elphmod.bravais.reciprocals(*a))
    B = np.array(elphmod.bravais.reciprocals(*A))

    nq = 100

    q = np.array([q1 * B[0] + q2 * B[1]
        for q1 in range(-nq, nq + 1)
        for q2 in range(-nq, nq + 1)])

    maxproj = 0.5 * np.dot(b[0], b[0]) + 1e-8

    q = np.array([qi for qi in q if
        maxproj >= abs(np.dot(qi, b[0])) and
        maxproj >= abs(np.dot(qi, b[1])) and
        maxproj >= abs(np.dot(qi, b[0] - b[1]))])

    # q are the Gamma points of the SC BZ within the UC BZ

    S = abs(np.exp(-2j * np.pi * q.dot(R.T)).sum(axis=-1)) ** 2
    S /= len(R) ** 2

    logS_min = min(np.log10(S).min(), logS_min)
    logS_max = max(np.log10(S).max(), logS_max)

    q = elphmod.bravais.rotate(q.T, phi, two_dimensional=False).T

    q *= 84

    bzpos = ((0.5 + 5 / (18 * 6)) * np.linalg.norm(A[0]),
        6.5 * np.linalg.norm(a[0]))

    q[:, :2] += bzpos

    ax.scatter(*bzpos, c='white', s=27500, marker='h', linewidth=0, zorder=5)

    bz = ax.scatter(q[:, 0], q[:, 1], c=S, s=27.5, marker='H', linewidth=0,
        cmap='cubehelix', norm=matplotlib.colors.LogNorm(vmin=1e-15, vmax=1),
        aa=False, zorder=5)

    if label == 'polaron248':
        cb = fig.add_axes(rect=(0.045, 0.6, 0.0075, 0.3))
        fig.colorbar(bz, cax=cb)
        cb.yaxis.set_ticks_position('left')
        cb.set_title('$S$')

    u = R - R0

    tau = np.linalg.norm(R0[0, :2] - R0[1, :2])

    for na in range(len(R)):
        if np.isclose(R0[na] @ A[0],
                np.linalg.norm(R0[na]) * np.linalg.norm(A[0])):

            R0[na] += A[1]

        if R0[na] @ A[0] < 1e-5:
            R0[na] += A[0]

        if R0[na] @ A[0] > 0.75 * np.linalg.norm(A[0]) ** 2 - 1e-5:
            R0[na] -= A[0]

        u[na] = elphmod.bravais.rotate(u[na], phi, two_dimensional=False)
        R0[na] = elphmod.bravais.rotate(R0[na], phi, two_dimensional=False)

    R = R0 + u

    bonds = np.array(storylines.bonds(R1=R[0::3, :2], R2=R[1::3, :2],
        dmin=0.1 * tau, dmax=1.5 * tau))

    lengths = np.linalg.norm(bonds[:, 1] - bonds[:, 0], axis=1)

    length_min = min(lengths.min(), length_min)
    length_max = max(lengths.max(), length_max)

    normalize = matplotlib.colors.Normalize(1.71356058310615, 2.01285683471413)

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

    arrow = {'->': True}

    ok = np.linalg.norm(u[:, :2], axis=1) > 0.035

    ax.quiver(*R[ok, :2].T, *19 * u[ok, :2].T, angles='xy', scale_units='xy',
        scale=1, width=0.006, headwidth=3, headlength=2, headaxislength=2,
        zorder=3)

    ax.text(-0.06, 0.94, '(%s)' % abc, transform=ax.transAxes)

print(logS_min, logS_max)
print(length_min, length_max)

fig.savefig('plot.pdf')

storylines.rasterize('plot', width=3500)

image = np.array(storylines.load('plot.png'))[:, :, :3]

storylines.save('plot.png', image)

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
