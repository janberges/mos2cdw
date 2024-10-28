#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines
import sys
import matplotlib.pyplot as plt

xyz = sys.argv[1]

comm = elphmod.MPI.comm
info = elphmod.MPI.info

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

a = elphmod.bravais.primitives(**pw, bohr=True)
b = np.array(elphmod.bravais.reciprocals(*a))

if comm.rank == 0:
    with open(xyz) as lines:
        next(lines)
        A = np.reshape(list(map(float, next(lines).split()[-9:])), (3, 3)).T.copy()
        N = np.round(A.dot(b.T)).astype(int)
else:
    N = np.empty((3, 3), dtype=int)

comm.Bcast(N)

info('Plotting %s x %s x %s supercell' % tuple(map(tuple, N)))

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph.clear()

driver = elphmod.md.Driver(elph,
    n=2 - pw['tot_charge'],
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
    supercell=N,
    unscreen=False,
)

driver.from_xyz(xyz)

r = driver.elph.ph.r
u = driver.u.reshape((-1, 3))
a = driver.elph.ph.a
R = r + u

b0 = np.array(elphmod.bravais.reciprocals(*ph.a))
b = np.array(elphmod.bravais.reciprocals(*a))

nq = 55
q = np.array([q1 * b[0] + q2 * b[1] for q1 in range(-nq, nq + 1) for q2 in range(-nq, nq + 1)])

M1 = 54 * b[0]
M2 = 54 * b[1]
M3 = 54 * b[0] - 54 * b[1]

q = np.array([qxy for qxy in q
    if abs(np.dot(qxy, M1)) / np.dot(M1, M1) <= 0.5 + 1e-8
    and abs(np.dot(qxy, M2)) / np.dot(M2, M2) <= 0.5 + 1e-8
    and abs(np.dot(qxy, M3)) / np.dot(M3, M3) <= 0.5 + 1e-8])

S = abs(np.exp(-2j * np.pi * q.dot(R.T)).sum(axis=-1)) ** 2
S = np.log(S)

if comm.rank == 0:
    plt.scatter(q[:, 0], q[:, 1], c=S, s=20, marker='h')
    #plt.scatter(b0[:, 0], b0[:, 1], color='yellow', s=20, marker='h')
    plt.axis('image')
    plt.axis('off')
    plt.savefig(xyz.replace('.xyz', '.png'))

