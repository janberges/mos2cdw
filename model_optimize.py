#!/usr/bin/env python3

import elphmod
import model
import numpy as np
import scipy.optimize

nk = 48

q = np.array([[0.0, 0.0], [0.0, np.pi], [2 * np.pi / 3, 2 * np.pi / 3]])

el = elphmod.el.Model('data/MoS2_3')
ph = elphmod.ph.Model('data/MoS2.ifc', apply_asr_simple=True)
elph = elphmod.elph.Model('data/MoS2_3.epmatwp', 'data/MoS2_3.wigner', el, ph)

g0 = elph.sample(q, (nk, nk))

def error(t):
    coupling = model.setup_coupling(*t)

    g = elphmod.elph.sample(coupling, q, (nk, nk))

    error = (abs(g - g0) ** 2).sum()

    print(('%9.5f' * 7) % (error, *t))

    return error

t = scipy.optimize.minimize(error,
    [-0.31479, 0.54522, -0.21612, -0.33978, 0.13760, 0.05757]).x

coupling = model.setup_coupling(*t)

model.save_coupling('data/model', coupling, el, ph)
