#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

phonons = False

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('dft/MoS2.epmatwp', 'dft/MoS2.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(2, 2, shared_memory=True)

driver = elphmod.md.Driver(elph, nk=(12, 12), nq=(6, 6) if phonons else (1, 1),
    n=(2 - pw['tot_charge']) * len(elph.cells),
    kT=pw['degauss'], f=elphmod.occupations.smearing(pw['smearing']))

driver.n = 2.2 * len(elph.cells)
driver.kT = 0.01

driver.random_displacements()

driver.plot(scale=10.0, interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(label=True, interactive=False)

driver.to_xyz('relaxed.xyz')

if phonons:
    ph = driver.phonons()

    path = 'GMKG'
    q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)

    w2 = elphmod.dispersion.dispersion(ph.D, q)

    if elphmod.MPI.comm.rank == 0:
        w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

        plt.plot(x, w, 'k')
        plt.ylabel('Phonon energy (meV)')
        plt.xlabel('Wave vector')
        plt.xticks(x[corners], path)
        plt.show()
