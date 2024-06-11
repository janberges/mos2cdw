#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

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
    supercell=(12, 12)
)

driver.n = 2.4 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.random_displacements()

driver.plot(scale=10.0, interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(interactive=False)

driver.to_xyz('relax_large.xyz')
