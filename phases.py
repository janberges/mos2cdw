#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

comm = elphmod.MPI.comm
info = elphmod.MPI.info

dopings = np.linspace(0.0, 0.6, 61)

pw = elphmod.bravais.read_pwi('data/MoS2.pwi')

el = elphmod.el.Model('data/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('data/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('data/model.epmatwp', 'data/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(2, 2, shared_memory=True)

cells = len(elph.cells)

driver = elphmod.md.Driver(elph,
    nk=(12, 12),
    nq=(12, 12),
    n=(2 - pw['tot_charge']) * cells,
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.diagonalize()
mu0 = driver.mu

driver.plot(scale=10.0, interactive=True)

if comm.rank == 0:
    data = open('phases.dat', 'w')
    data.write((' %9s' * 10 + '\n') % ('nel', 'xel',
        'dE/eV', 'mu/eV', '|u|/AA',
        'lamda', 'wlog/eV', 'w2nd/eV', 'wmin/eV', 'Tc/K'))

for doping in dopings:
    info('Setting the doping to %g electrons per unit cell...' % doping)

    driver.n = (2 + doping) * cells

    driver.u[:] = 0.0

    E0 = driver.free_energy()

    driver.random_displacements()

    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

    E = driver.free_energy()

    lamda, wlog, w2nd, wmin = driver.superconductivity()

    wlog *= elphmod.misc.Ry
    w2nd *= elphmod.misc.Ry
    wmin *= elphmod.misc.Ry

    Tc = elphmod.eliashberg.Tc(lamda, wlog, 0.0, w2nd, correct=True)

    if comm.rank == 0:
        data.write((' %9.6f' * 10 + '\n') % (driver.n, doping,
            (E - E0) * elphmod.misc.Ry / cells,
            (driver.mu - mu0) * elphmod.misc.Ry,
            np.linalg.norm(driver.u) * elphmod.misc.a0 / np.sqrt(cells),
            lamda, wlog, w2nd, wmin, Tc))

    info('The critical temperature is %g K.' % Tc)

if comm.rank == 0:
    data.close()
