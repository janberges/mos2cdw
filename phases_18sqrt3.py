#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize
import sys

nel = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ini = str(sys.argv[2]) if len(sys.argv) > 2 else '2x2'

mustar = 0.13

pw = elphmod.bravais.read_pwi('data/MoS2.pwi')

el = elphmod.el.Model('data/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('data/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('data/model.epmatwp', 'data/model.wigner',
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

def cdw_2x2(distortion=0.02, amplitude=0.1):
    driver.random_displacements(distortion)
    random = driver.u.copy()

    symmetric()

    dim = 3 * driver.elph.ph.nat // cells
    u = amplitude * np.array([1.0, 0.0])

    for n, (i, j, k) in enumerate(driver.elph.cells):
        for I, J, twelfth in (1, 0, 3), (0, 1, 7), (1, 1, 11):
            if i % 2 == I and j % 2 == J:
                driver.u[n * dim + 3:n * dim + 5] = elphmod.bravais.rotate(u,
                    twelfth * np.pi / 6)

    driver.u += random

symmetric()
driver.to_xyz('phases_18sqrt3/symmetric.xyz')

driver.diagonalize()
mu0 = driver.mu

driver.n = 2 * cells + nel
E0 = driver.free_energy()

if '2x2' in ini:
    cdw_2x2()
elif 'triangle' in ini:
    triangle()

scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
    method='BFGS', options=dict(gtol=1e-5, norm=np.inf))

driver.to_xyz('phases_18sqrt3/dop_%03d_from_%s.xyz' % (nel, ini))

E = driver.free_energy()

N0 = driver.f.delta(driver.e / driver.kT).sum() / driver.kT

lamda, wlog, w2nd, wmin = driver.superconductivity()

wlog *= elphmod.misc.Ry
w2nd *= elphmod.misc.Ry
wmin *= elphmod.misc.Ry

Tc = elphmod.eliashberg.Tc(lamda, wlog, mustar, w2nd, correct=True)

if elphmod.MPI.comm.rank == 0:
    with open('phases_18sqrt3/dop_%03d_from_%s.dat' % (nel, ini), 'w') as data:
        data.write(('%3s' + ' %9s' * 10 + '\n') % ('nel', 'xel',
            'dE/eV', 'N0*eV', 'mu/eV', '|u|/AA',
            'lamda', 'wlog/eV', 'w2nd/eV', 'wmin/eV', 'Tc/K'))

        data.write(('%3d' + ' %9.6f' * 10 + '\n') % (nel, nel / cells,
            (E - E0) * elphmod.misc.Ry / cells,
            N0 / elphmod.misc.Ry / cells,
            (driver.mu - mu0) * elphmod.misc.Ry,
            np.linalg.norm(driver.u) * elphmod.misc.a0 / np.sqrt(cells),
            lamda, wlog, w2nd, wmin, Tc))
