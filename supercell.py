#!/usr/bin/env python3

import elphmod
import numpy as np

info = elphmod.MPI.info

ph = elphmod.ph.Model('dft/MoS2.ifc')
ph.clear()

ph_1 = ph.supercell((24, 12, 0), (-12, 12, 0))
ph_2 = ph.supercell((6, 3, 0), (-3, 3, 0))
ph_3 = ph.supercell(2, 2)

info('%g' % (np.linalg.norm(ph_1.a[0]) / np.linalg.norm(ph.a[0]) / np.sqrt(3)))
info('%g' % (np.linalg.norm(ph_2.a[0]) / np.linalg.norm(ph.a[0]) / np.sqrt(3)))
info('%g' % (len(ph_1.cells) / len(ph_2.cells)))
