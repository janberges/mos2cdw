#!/usr/bin/env python3

import elphmod
import numpy as np
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

minimum = +np.inf
maximum = -np.inf

for xyz in sys.argv[1:]:
    info('Processing %s' % xyz)

    driver.from_xyz(xyz)

    R = driver.elph.ph.r + driver.u.reshape((-1, 3))

    b = np.array(elphmod.bravais.reciprocals(*ph.a))
    B = np.array(elphmod.bravais.reciprocals(*driver.elph.ph.a))

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
    S /= driver.elph.ph.nat ** 2

    minimum = min(np.log(S).min(), minimum)
    maximum = max(np.log(S).max(), maximum)

    if comm.rank == 0:
        plt.close()
        plt.scatter(q[:, 0], q[:, 1], c=np.log(S), s=68, marker='h',
            linewidth=0, aa=False, vmin=-50.359216312684325, vmax=0.0)
            #, cmap='plasma')
        plt.axis('image')
        plt.axis('off')
        plt.savefig(xyz.replace('.xyz', '_ft.png'), bbox_inches='tight')

print(minimum, maximum)
