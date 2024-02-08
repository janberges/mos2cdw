#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

phonons = False
triangle = True

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

if triangle:
    atoms = [1, 4, 10]

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

driver.plot(label=True, interactive=False)

driver.to_xyz('relaxed.xyz')

if phonons:
    ph = elphmod.ph.Model('dft/MoS2.ifc', apply_asr_simple=True)
    Ph = driver.phonons(apply_asr_simple=True)

    path = 'GMKG'
    q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)
    Q = np.dot(np.dot(q, elphmod.bravais.reciprocals(*ph.a)), Ph.a.T)

    w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
    W2, U = elphmod.dispersion.dispersion(Ph.D, q, vectors=True)

    W = elphmod.dispersion.unfolding_weights(q, Ph.cells, u, U)
    W = np.ones(W2.shape)

    linewidth = 1.0

    if elphmod.MPI.comm.rank == 0:
        for nu in range(W2.shape[1]):
            fatband, = elphmod.plot.compline(x, elphmod.ph.sgnsqrt(W2[:, nu])
                * 1e3 * elphmod.misc.Ry, linewidth * W[:, nu])

            plt.fill(*fatband, linewidth=0.0, color='firebrick')

        plt.ylabel('Phonon energy (meV)')
        plt.xlabel('Wave vector')
        plt.xticks(x[corners], path)
        plt.show()
