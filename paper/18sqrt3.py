#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize
import sys

comm = elphmod.MPI.comm
info = elphmod.MPI.info

pw = elphmod.bravais.read_pwi('../dft/MoS2.pwi')

el = elphmod.el.Model('../dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('../dft/MoS2.ifc', divide_mass=False,
    apply_asr_simple=True)
elph = elphmod.elph.Model('../model/model.epmatwp', '../model/model.wigner',
    el, ph, divide_mass=False, shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=pw['k_points'][:3],
    nq=ph.nq,
    n=2 - pw['tot_charge'],
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
    supercell=[(36, 18, 0), (-18, 18, 0)],
)

cells = len(driver.elph.cells)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

def symmetric():
    driver.u[:] = 0.0

def triangle(atoms=[1471, 1474, 1555], amplitude=0.2):
    symmetric()

    center = np.average(driver.elph.ph.r[atoms], axis=0)

    for atom in atoms:
        u = center - driver.elph.ph.r[atom]
        u *= amplitude / np.linalg.norm(u)

        driver.u[3 * atom:3 * atom + 3] = u

def triangles(distortion=0.02, amplitude=0.1):
    driver.random_displacements(distortion)
    random = driver.u.copy()

    symmetric()

    dim = 3 * driver.elph.ph.nat // cells
    u = amplitude * np.array([1.0, 0.0])

    for n, (i, j, k) in enumerate(driver.elph.cells):
        for I, J, twelfth in (1, 1, 3), (0, 0, 7), (1, 0, 11):
            if i % 2 == I and j % 2 == J:
                driver.u[n * dim + 3:n * dim + 5] = elphmod.bravais.rotate(u,
                    twelfth * np.pi / 6)

    driver.u += random

def optimize():
    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', options=dict(gtol=1e-5, norm=np.inf))

driver.diagonalize()

mu0 = driver.mu

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
step = int(sys.argv[3]) if len(sys.argv) > 3 else 1

for nel in range(start, stop, step):
    info('Number of doping electrons: %g' % nel)

    driver.n = 2 * cells + nel

    symmetric()

    E0 = driver.free_energy()

    triangle()
    optimize()
    driver.to_xyz('polaron%03d.xyz' % nel)

    #triangles()
    #optimize()
    #driver.to_xyz('cdw%03d.xyz' % nel)

    E = driver.free_energy()

    lamda, wlog, w2nd = driver.superconductivity()

    wlog *= elphmod.misc.Ry
    w2nd *= elphmod.misc.Ry

    Tc = elphmod.eliashberg.Tc(lamda, wlog, 0.0, w2nd, correct=True)

    info('The critical temperature is %g K.' % Tc)

    if comm.rank != 0:
        continue

    with open('polaron%03d.dat' % nel, 'w') as data:
        data.write(('%3s' + ' %9s' * 8 + '\n') % ('nel', 'xel',
            'dE/eV', 'mu/eV', '|u|/AA', 'lamda', 'wlog/eV', 'w2nd/eV', 'Tc/K'))

        data.write(('%3d' + ' %9.6f' * 8 + '\n') % (nel, nel / cells,
            (E - E0) * elphmod.misc.Ry / cells,
            (driver.driver.mu - mu0) * elphmod.misc.Ry,
            np.linalg.norm(driver.u) * elphmod.misc.a0 / np.sqrt(cells)),
            lamda, wlog, w2nd, Tc)
