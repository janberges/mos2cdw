#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

info = elphmod.MPI.info

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
    supercell=[(24, 12, 0), (-12, 12, 0)],
)

#driver.plot(label=True, scale=20.0, interactive=False)

driver.n = 2.3 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.random_displacements()

#atoms = [598, 601, 655]
#
#center = np.average(driver.elph.ph.r[atoms], axis=0)
#
#for atom in atoms:
#    u = center - driver.elph.ph.r[atom]
#    u *= 0.3 / np.linalg.norm(u)
#
#    driver.u[3 * atom:3 * atom + 3] = u

#driver.from_xyz('relax_large.xyz')

driver.plot(scale=20.0, interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(interactive=False)

driver.to_xyz('relax_large.xyz')
