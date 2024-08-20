#!/usr/bin/env python3 -i

import elphmod
import numpy as np
import scipy.optimize
import sys

comm = elphmod.MPI.comm
info = elphmod.MPI.info

dopings = np.linspace(0.0, 0.6, 61)

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=pw['k_points'][:3],
    nq=ph.nq,
    n=2 - pw['tot_charge'],
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
    supercell=[(24, 12, 0), (-12, 12, 0)],
)

#driver.plot(label=True, scale=20.0, interactive=False)

#driver.n = 2.3 * len(driver.elph.cells)
driver.kT = 0.005
driver.f = elphmod.occupations.fermi_dirac

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

def random():
    driver.random_displacements()

def triangle(atoms=[658, 661, 712], amplitude=0.2):
    driver.u[:] = 0.0

    center = np.average(driver.elph.ph.r[atoms], axis=0)

    for atom in atoms:
        u = center - driver.elph.ph.r[atom]
        u *= amplitude / np.linalg.norm(u)

        driver.u[3 * atom:3 * atom + 3] = u

def relaxed():
    driver.from_xyz('relax_large.xyz')

def optimize():
    scipy.optimize.minimize(driver.free_energy, driver.u, jac=driver.jacobian,
        method='BFGS', options=dict(gtol=1e-6, norm=np.inf))

driver.plot(scale=20.0, interactive=True)

raise SystemExit

# 12 sqrt(3):
#triangle(atoms=[658, 661, 712], amplitude=0.2)
#triangle(atoms=[601, 655, 658], amplitude=-0.2)

# 18 sqrt(3):
#triangle(atoms=[1471, 1474, 1555], amplitude=0.2)
#triangle(atoms=[1390, 1468, 1471], amplitude=-0.2)

# 18:
#triangle(atoms=[457, 460, 514], amplitude=0.2)
#triangle(atoms=[514, 568, 571], amplitude=-0.2)

# 27:
#triangle(atoms=[1009, 1012, 1093], amplitude=0.2)
#triangle(atoms=[1093, 1174, 1177], amplitude=-0.2)

#optimize()
#
#driver.plot(interactive=False)
#
#driver.to_xyz('relax_large.xyz')

if len(sys.argv) > 1:
    idoping = int(sys.argv[1])
    dopings = dopings[idoping - 1::6]
    filename = 'relax_large_%d.dat' % idoping
else:
    filename = 'relax_large.dat'

if comm.rank == 0:
    data = open(filename, 'w')
    data.write(' %9s' * 6
        % ('doping/e', 'lamda', 'wlog/eV', 'w2nd/eV', 'Tc/K', '|u|/AA'))
    data.write('\n')

for doping in dopings:
    info('Setting the doping to %g electrons per unit cell...' % doping)

    driver.n = (2 + doping) * len(driver.elph.cells)

    #relaxed()
    #triangles()
    triangle()

    optimize()

    driver.to_xyz('relax_large_%.2f.xyz' % doping)

    driver.diagonalize()

    lamda, wlog, w2nd = driver.superconductivity(kT=0.01)

    wlog *= elphmod.misc.Ry
    w2nd *= elphmod.misc.Ry

    Tc = elphmod.eliashberg.Tc(lamda, wlog, 0.0, w2nd, correct=True)

    if comm.rank == 0:
        data.write(' %9.6f' * 6 % (doping, lamda, wlog, w2nd, Tc,
            np.linalg.norm(driver.u) * elphmod.misc.a0
                * np.sqrt(4 / len(driver.elph.cells))))
        data.write('\n')

    info('The critical temperature is %g K.' % Tc)

if comm.rank == 0:
    data.close()
