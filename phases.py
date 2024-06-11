#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

comm = elphmod.MPI.comm
info = elphmod.MPI.info

nk = nq = 12
kTel = 0.01
f = elphmod.occupations.fermi_dirac
eps = 1e-10

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

q = sorted(elphmod.bravais.irreducibles(nq))
q = 2 * np.pi / nq * np.array(q, dtype=float)

g = driver.elph.sample(q, nk=nk, shared_memory=True)

node, images = elphmod.MPI.shm_split()

if comm.rank == 0:
    g /= np.repeat(np.sqrt(driver.elph.ph.M), 3)[np.newaxis, :,
        np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    g *= elphmod.misc.Ry ** 1.5

if node.rank == 0:
    images.Bcast(g)

kTel *= elphmod.misc.Ry

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

    ph = driver.phonons(apply_asr_simple=True)

    w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
    w2 *= elphmod.misc.Ry ** 2

    if np.any(w2 < -1e-8):
        info('Imaginary frequencies!')
        continue

    el = driver.electrons()

    e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)
    e = e[..., el.size // 3:]
    U = U[..., el.size // 3:]

    g2 = elphmod.elph.transform(g, q, nk, U, u, squared=True,
        shared_memory=True)

    lamda, wlog, Tc, w2nd = elphmod.eliashberg.McMillan(nq, e, w2, g2, eps,
        mustar=0.0, kT=kTel, f=f, correct=True)

    if comm.rank == 0:
        data.write(' %9.6f' * 6 % (doping, lamda, wlog, w2nd, Tc,
            np.linalg.norm(driver.u) * elphmod.misc.a0))
        data.write('\n')

    info('The critical temperature is %g K.' % Tc)

if comm.rank == 0:
    data.close()
