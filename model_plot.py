#!/usr/bin/env python3

import elphmod
import numpy as np
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

q = np.array([[0.0, np.pi]])

BZ = dict(points=100, outside=np.nan)

el = elphmod.el.Model('data/MoS2_3')
ph = elphmod.ph.Model('data/MoS2.ifc', apply_asr_simple=True)
elph = elphmod.elph.Model('data/MoS2_3.epmatwp', 'data/MoS2_3.wigner', el, ph)
model = elphmod.elph.Model('data/model.epmatwp', 'data/model.wigner', el, ph)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g02 = elph.sample(q, (nk, nk), u=u[..., 1:2], squared=True)
g2 = model.sample(q, (nk, nk), u=u[..., 1:2], squared=True)

g2max = max(g02.max(), g2.max())

for name, coupling in ('model_plot_mod', g2), ('model_plot_ref', g02):
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
