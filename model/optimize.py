#!/usr/bin/env python3

import elphmod
import model
import numpy as np
import scipy.optimize
import storylines

comm = elphmod.MPI.comm

nk = 48

AFMhot = storylines.colormap( # Gnuplot
    (0.00, storylines.Color(0, 0, 0)),
    (0.25, storylines.Color(128, 0, 0)),
    (0.50, storylines.Color(255, 128, 0)),
    (0.75, storylines.Color(255, 255, 128)),
    (1.00, storylines.Color(255, 255, 255)),
    )

q = np.array([[0.0, 0.0], [0.0, np.pi], [2 * np.pi / 3, 2 * np.pi / 3]])

BZ = dict(points=100, outside=np.nan)

el = elphmod.el.Model('../dft/MoS2_3')
ph = elphmod.ph.Model('../dft/MoS2.ifc', apply_asr_simple=True)
elph = elphmod.elph.Model('../dft/MoS2_3.epmatwp', '../dft/MoS2_3.wigner', el, ph)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g0 = elph.sample(q, (nk, nk))
g02 = abs(g0) ** 2

def error(t):
    coupling = model.setup_coupling(*t)

    g = elphmod.elph.sample(coupling, q, (nk, nk))

    error = (abs(g - g0) ** 2).sum()

    print(('%9.5f' * 7) % (error, *t))

    return error

t = scipy.optimize.minimize(error,
    [0.31479, -0.54522, 0.21612, 0.33978, -0.13760, -0.05757]).x

coupling = model.setup_coupling(*t)

model.save_coupling('model', coupling, el, ph)

g0 = elph.sample(q[1:2], (nk, nk), u=u[1:2, :, 1:2])
g02 = abs(g0) ** 2

g = elphmod.elph.sample(coupling, q[1:2], (nk, nk), u=u[1:2, :, 1:2])
g2 = abs(g) ** 2

g2max = max(g02.max(), g2.max())

for name, coupling in ('coupling_mod', g2), ('coupling_ref', g02):
    for a in range(el.size):
        for b in range(el.size):
            data = elphmod.plot.toBZ(coupling[0, 0, :, :, a, b], **BZ)

            image = storylines.colorize(data, cmap=AFMhot,
                minimum=0, maximum=g2max)

            if comm.rank == 0:
                plot = storylines.Plot(background='%s%d%d.png' % (name, a, b),
                    width=1.0, height=0, xyaxes=False, margin=0.05)

                plot.line(*zip(*elphmod.bravais.BZ()), thick=True)

                storylines.save(plot.background, image)

                plot.save('%s%d%d.pdf' % (name, a, b))

    if comm.rank == 0:
        storylines.combine('%s.pdf' % name, ['%s%d%d' % (name, a, b)
            for a in range(el.size) for b in range(el.size)], columns=el.size)
