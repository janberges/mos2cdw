#!/usr/bin/env python3

import elphmod
import numpy as np

el = elphmod.el.Model('dft/MoS2')
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False)
elph = elphmod.elph.Model('dft/MoS2.epmatwp', 'dft/MoS2.wigner', el, ph,
    divide_mass=False)

select = [0, 3, 4]
el.size = len(select)
el.data = el.data[:, select, :]
el.data = el.data[:, :, select]
elph.data = elph.data[:, :, :, select, :]
elph.data = elph.data[:, :, :, :, select]

el.to_hrdat('dft/MoS2_3')
elph.to_epmatwp('dft/MoS2_3')
