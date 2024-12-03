#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

pw = elphmod.bravais.read_pwi('data/MoS2.pwi')

el = elphmod.el.Model('data/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('data/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('data/model.epmatwp', 'data/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(2, 2, shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=(12, 12),
    nq=(6, 6),
    n=(2 - pw['tot_charge']) * len(elph.cells),
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
)

driver.n = 2.4 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.random_displacements()

driver.plot(scale=10.0, interactive=True)

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(interactive=False)

ph = elphmod.ph.Model('data/MoS2.ifc', apply_asr_simple=True)
Ph = driver.phonons(apply_asr_simple=True)

path = 'GMKG'
q, x, corners = elphmod.bravais.path(path, ibrav=4, N=150)
Q = np.dot(np.dot(q, elphmod.bravais.reciprocals(*ph.a)), Ph.a.T)

figure, ax = plt.subplots(1, 2, sharey='row')

W2 = elphmod.dispersion.dispersion(Ph.D, q)

if elphmod.MPI.comm.rank == 0:
    ax[0].plot(x, elphmod.ph.sgnsqrt(W2) * 1e3 * elphmod.misc.Ry,
        color='firebrick')

    ax[0].set_title('On supercell BZ')
    ax[0].set_ylabel('Phonon energy (meV)')
    ax[0].set_xlabel('Wave vector')
    ax[0].set_xticks(x[corners], [X + "$'$" for X in path])

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
W2, U = elphmod.dispersion.dispersion(Ph.D, Q, vectors=True)

weight = elphmod.dispersion.unfolding_weights(q, Ph.cells, u, U)

if elphmod.MPI.comm.rank == 0:
    for nu in range(W2.shape[1]):
        fatband, = elphmod.plot.compline(x, elphmod.ph.sgnsqrt(W2[:, nu])
            * 1e3 * elphmod.misc.Ry, 0.5 * weight[:, nu])

        ax[1].fill(*fatband, linewidth=0.0, color='firebrick')

    ax[1].set_title('Unfolded to original BZ')
    ax[1].set_ylabel('Phonon energy (meV)')
    ax[1].set_xlabel('Wave vector')
    ax[1].set_xticks(x[corners], path)

plt.show()
