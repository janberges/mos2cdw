#!/usr/bin/env python3

import elphmod

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph = elph.supercell(2, 2, shared_memory=True)

# instead, for 2 sqrt(3) x 2 sqrt(3) cell (also reduce nk and nq):
#elph = elph.supercell((4, 2, 0), (-2, 2, 0), shared_memory=True)

driver = elphmod.md.Driver(elph,
    nk=(12, 12),
    nq=(6, 6),
    n=(2 - pw['tot_charge']) * len(elph.cells),
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
)

driver.save('driver_small.pickle')
