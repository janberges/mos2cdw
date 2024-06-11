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

driver = elphmod.md.Driver.load('driver_small.pickle')

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

def set_triangle(driver, atoms=[1, 4, 10], displacement=0.2):
    center = np.average(driver.elph.ph.r[atoms], axis=0)

    for atom in atoms:
        u = center - driver.elph.ph.r[atom]
        u *= displacement / np.linalg.norm(u)

        driver.u[3 * atom:3 * atom + 3] = u

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

    for cdw in False, True:
        if cdw:
            info('Checking for a CDW...')

            set_triangle(driver)

            scipy.optimize.minimize(driver.free_energy, driver.u,
                jac=driver.jacobian, method='BFGS',
                options=dict(gtol=1e-6, norm=np.inf))

            if np.all(abs(driver.u) < 1e-3):
                info('No CDW!')
                continue
        else:
            info('Checking the undistorted system...')

            driver.u[:] = 0.0

        driver.diagonalize()

        ph = driver.phonons(apply_asr_simple=True)

        w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)
        w2 *= elphmod.misc.Ry ** 2

        if np.any(w2[1:] < 0.0):
            info('Imaginary frequencies!')
            continue

        el = driver.electrons()

        e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)

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
