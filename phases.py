#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

comm = elphmod.MPI.comm
info = elphmod.MPI.info

dopings = np.linspace(0.0, 0.6, 61)

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(2, 2, shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=(12, 12),
    nq=(12, 12),
    n=(2 - pw['tot_charge']) * len(elph.cells),
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

driver.plot(scale=10.0, interactive=True)

if comm.rank == 0:
    data = open('phases.dat', 'w')
    data.write(' %9s' * 6
        % ('doping/e', 'lamda', 'wlog/eV', 'w2nd/eV', 'Tc/K', '|u|/AA'))
    data.write('\n')

for doping in dopings:
    info('Setting the doping to %g electrons per unit cell...' % doping)

    driver.n = (2 + doping) * len(driver.elph.cells)

    driver.random_displacements()

    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

    if np.all(abs(driver.u) < 1e-3):
        info('No CDW!')

    driver.diagonalize()

    lamda, wlog, w2nd = driver.superconductivity(kT=0.01)

    if lamda is None:
        info('Imaginary frequencies!')
        continue

    wlog *= elphmod.misc.Ry
    w2nd *= elphmod.misc.Ry

    Tc = elphmod.eliashberg.Tc(lamda, wlog, 0.0, w2nd, correct=True)

    if comm.rank == 0:
        data.write(' %9.6f' * 6 % (doping, lamda, wlog, w2nd, Tc,
            np.linalg.norm(driver.u) * elphmod.misc.a0))
        data.write('\n')

    info('The critical temperature is %g K.' % Tc)

if comm.rank == 0:
    data.close()
