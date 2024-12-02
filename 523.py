#!/usr/bin/env python3

import elphmod

el = elphmod.el.Model('data/MoS2')
ph = elphmod.ph.Model('data/MoS2.ifc', divide_mass=False)
elph = elphmod.elph.Model('data/MoS2.epmatwp', 'data/MoS2.wigner', el, ph,
    divide_mass=False)

select = [0, 3, 4]
el.size = len(select)
el.data = el.data[:, select, :]
el.data = el.data[:, :, select]
elph.data = elph.data[:, :, :, select, :]
elph.data = elph.data[:, :, :, :, select]

el.to_hrdat('data/MoS2_3')
elph.to_epmatwp('data/MoS2_3')
