#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

N = 12
triangles = True

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=pw['k_points'][:3],
    nq=ph.nq,
    n=2 - pw['tot_charge'],
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
    supercell=(N, N)
)

driver.n = 2.4 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

if triangles:
    for row in range(0, N, 2):
        for col in range(0, N, 2):
            at1 = 3 * (row + N * col) + 1
            at2 = at1 + 3
            at3 = at2 + 3 * N

            atoms = [at1, at2, at3]

            center = np.average(driver.elph.ph.r[atoms], axis=0)

            for atom in atoms:
                u = center - driver.elph.ph.r[atom]
                u *= 0.2 / np.linalg.norm(u)

                driver.u[3 * atom:3 * atom + 3] = u
else:
    driver.random_displacements()

driver.plot(scale=10.0, interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(interactive=False)

driver.to_xyz('relax_large.xyz')
