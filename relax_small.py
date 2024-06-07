#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

phonons = True
unfold = False
triangle = True
linewidth = 0.5

driver = elphmod.md.Driver.load('driver_small.pickle')

driver.n = 2.3 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

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
    Ph = driver.phonons(apply_asr_simple=True)

    path = 'GMKG'
    q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)

    if unfold:
        ph = elphmod.ph.Model('dft/MoS2.ifc', apply_asr_simple=True)

        Q = np.dot(np.dot(q, elphmod.bravais.reciprocals(*ph.a)), Ph.a.T)

        w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
        W2, U = elphmod.dispersion.dispersion(Ph.D, Q, vectors=True)

        W = elphmod.dispersion.unfolding_weights(q, Ph.cells, u, U)
    else:
        W2, U = elphmod.dispersion.dispersion(Ph.D, q, vectors=True)

        W = np.ones(W2.shape)

    if elphmod.MPI.comm.rank == 0:
        for nu in range(W2.shape[1]):
            fatband, = elphmod.plot.compline(x, elphmod.ph.sgnsqrt(W2[:, nu])
                * 1e3 * elphmod.misc.Ry, linewidth * W[:, nu])

            plt.fill(*fatband, linewidth=0.0, color='firebrick')

        plt.ylabel('Phonon energy (meV)')
        plt.xlabel('Wave vector')
        plt.xticks(x[corners], path)
        plt.show()
