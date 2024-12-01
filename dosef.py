#!/usr/bin/env python3

import elphmod
import os

comm = elphmod.MPI.comm
info = elphmod.MPI.info

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner',
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

for xyz in os.listdir('xyz'):
    if 'symmetric' in xyz or not xyz.endswith('.xyz'):
        continue

    driver.n = 2 * cells + int(xyz[-7:-4])

    driver.from_xyz('xyz/%s' % xyz)

    driver.diagonalize()

    dosef = driver.f.delta(driver.e / driver.kT).sum() / driver.kT

    info('%15s %9.6f' % (xyz, dosef / elphmod.misc.Ry / cells))
