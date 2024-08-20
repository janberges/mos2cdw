#!/usr/bin/env python3

import elphmod
import numpy as np
import scipy.optimize

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
    supercell=[(24, 12, 0), (-12, 12, 0)],
)

driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

def triangle(atoms, amplitude):
    driver.u[:] = 0.0

    center = np.average(driver.elph.ph.r[atoms], axis=0)

    for atom in atoms:
        u = center - driver.elph.ph.r[atom]
        u *= amplitude / np.linalg.norm(u)

        driver.u[3 * atom:3 * atom + 3] = u

def triangles():
    u = 0.1 * np.array([1.0, 0.0])

    driver.random_displacements(0.02)

    random = driver.u.copy()

    driver.u[:] = 0.0

    block = 3 * driver.elph.ph.nat // len(driver.elph.cells)

    for n, (i, j, k) in enumerate(driver.elph.cells):
        if i % 2 and j % 2:
            driver.u[n * block + 3:n * block + 5] = elphmod.bravais.rotate(u,
                3 * np.pi / 6)

        elif not i % 2 and not j % 2:
            driver.u[n * block + 3:n * block + 5] = elphmod.bravais.rotate(u,
                7 * np.pi / 6)

        elif i % 2 and not j % 2:
            driver.u[n * block + 3:n * block + 5] = elphmod.bravais.rotate(u,
                11 * np.pi / 6)

    driver.u += random

def optimize():
    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

for driver.n in np.arange(np.ceil(
    2.2 * len(driver.elph.cells)),
    2.4 * len(driver.elph.cells)):

    info('Number of electrons: %d' % driver.n)

    info('Inward triangle')

    triangle(atoms=[658, 661, 712], amplitude=0.2)
    optimize()
    driver.to_xyz('relax_12sqrt3_inward_%d.xyz' % driver.n)

    info('Outward triangle')

    triangle(atoms=[601, 655, 658], amplitude=-0.2)
    optimize()
    driver.to_xyz('relax_12sqrt3_outward_%d.xyz' % driver.n)

    info('Triangle CDW')

    triangles()
    optimize()
    driver.to_xyz('relax_12sqrt3_cdw_%d.xyz' % driver.n)
