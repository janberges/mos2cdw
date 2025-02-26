#!/usr/bin/env python3

import elphmod
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

pw = elphmod.bravais.read_pwi('data/MoS2.pwi')

a = elphmod.bravais.primitives(**pw)
b = np.array(elphmod.bravais.reciprocals(*a))

r0 = pw['r'].dot(a)

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

for xyz in sys.argv[1:]:
    A, typ, R = load_xyz(xyz)
    B = np.array(elphmod.bravais.reciprocals(*A))

    N = np.round(A.dot(b.T)).astype(int)

    cells = elphmod.bravais.supercell(*N)[-1]

    if elphmod.MPI.comm.rank != 0:
        continue

    print('Processing %s' % xyz)
    print('Detected %s x %s x %s supercell' % tuple(map(tuple, N)))

    R0 = np.array([n1 * a[0] + n2 * a[1] + n3 * a[2] + r0[na]
        for n1, n2, n3 in cells for na in range(len(r0))])

    tau = np.linalg.norm(R0[0, :2] - R0[1, :2])

    u = R - R0

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

    eps = 1e-5

    if np.diag(N).prod() != len(cells):
        eps *= -1

    phi = -np.arctan2(A[0, 1], A[0, 0])

    for na in range(len(R)):
        if np.isclose(R0[na] @ A[0],
                np.linalg.norm(R0[na]) * np.linalg.norm(A[0])):

            R0[na] += A[1]

        if R0[na] @ A[0] < -eps:
            R0[na] += A[0]

        if R0[na] @ A[0] > 0.75 * np.linalg.norm(A[0]) ** 2 + eps:
            R0[na] -= A[0]

        u[na] = elphmod.bravais.rotate(u[na], phi, two_dimensional=False)
        R0[na] = elphmod.bravais.rotate(R0[na], phi, two_dimensional=False)

    R = R0 + u

    fig, ax = plt.subplots()

    plt.scatter(R[1::3, 0], R[1::3, 1], s=20, color='#5f97e6', clip_on=False)
    plt.scatter(R[0::3, 0], R[0::3, 1], s=20, color='#d8ba8d', clip_on=False)

    ok = np.linalg.norm(u[:, :2], axis=1) > 0.03

    plt.quiver(*R[ok, :2].T, *20 * u[ok, :2].T, angles='xy', scale_units='xy',
        scale=1, minlength=0, clip_on=False)

    plt.xlim(R0[:, 0].min(), R0[:, 0].max())
    plt.ylim(R0[:, 1].min(), R0[:, 1].max())

    q = elphmod.bravais.rotate(q.T, phi, two_dimensional=False).T

    q *= tau / np.linalg.norm(B[0])

    q[:, 0] += R0[:, 0].max() - q[:, 0].min() + np.linalg.norm(a[0])
    q[:, 1] += R0[:, 1].min() - q[:, 1].min()

    ax.scatter(q[:, 0], q[:, 1], c=S, s=30, marker='H', linewidth=0,
        norm=matplotlib.colors.LogNorm(vmin=1e-15, vmax=1, clip=True),
        cmap='cubehelix')

    plt.axis('equal')
    plt.axis('off')

    fig.set_size_inches(0.05 * np.ptp(np.concatenate((q, R0)), axis=0)[:2])

    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)

    fig.savefig(xyz.replace('.xyz', '.pdf'))

    plt.close(fig)
