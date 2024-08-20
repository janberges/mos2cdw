#!/usr/bin/env python3

import elphmod
import numpy as np
import sys

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

driver.plot(label=False, scale=10.0, interactive=False)
