#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np
import storylines

comm = elphmod.MPI.comm

nk = 24
kT = 300.0 * elphmod.misc.kB / elphmod.misc.Ry
f = elphmod.occupations.fermi_dirac

q = np.array([[0.0, np.pi], [2 * np.pi / 3, 2 * np.pi / 3]])
q_label = 'MK'

n = 1
nu = [0, 1]

cmap = storylines.colormap(
    (0.00, storylines.Color(94, 60, 153)),
    (0.25, storylines.Color(153, 142, 195)),
    (0.75, storylines.Color(241, 163, 64)),
    (1.00, storylines.Color(230, 97, 1)),
    (None, storylines.Color(255, 255, 255)),
    )

BZ = dict(points=100, outside=np.nan)

el = elphmod.el.Model('../dft/mos2')
ph = elphmod.ph.Model('../dft/mos2.ifc', apply_asr_simple=True)
#elph = elphmod.elph.Model('../dft/mos2.epmatwp', '../dft/mos2.wigner',
#    el, ph)
elph = elphmod.elph.Model('../model/model.epmatwp', '../model/model.wigner',
    el, ph)

e, U = elphmod.dispersion.dispersion_full_nosym(el.H, nk, vectors=True)
e /= elphmod.misc.Ry
e -= elphmod.occupations.find_Fermi_level(2.15, e, kT, f)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

for iq in range(len(q)):
    for x in range(ph.size):
        u[iq, x, [0, nu[iq]]] = u[iq, x, [nu[iq], 0]]

g2 = abs(elph.sample(q, U=U[..., n:n + 1], u=u[..., :1])) ** 2

Pi = elphmod.diagrams.phonon_self_energy(q, e[..., n:n + 1], g2=g2[:, :1],
    kT=kT, occupations=f, fluctuations=True)[1]

X0 = elphmod.diagrams.phonon_self_energy(q, e[..., n:n + 1],
    kT=kT, occupations=f, fluctuations=True)[1]

for iq in range(len(q)):
    kxmax, kymax, kx, ky, ek1_BZ = elphmod.plot.toBZ(e[:, :, n], return_k=True,
        **BZ)

    ek2_BZ = elphmod.plot.toBZ(np.roll(np.roll(e[:, :, n],
        shift=-int(round(q[iq, 0] * nk / (2 * np.pi))), axis=0),
        shift=-int(round(q[iq, 1] * nk / (2 * np.pi))), axis=1), **BZ)

    Pi_BZ = -elphmod.plot.toBZ(Pi[iq, 0, :, :, 0, 0], **BZ)
    X0_BZ = -elphmod.plot.toBZ(X0[iq, 0, :, :, 0, 0], **BZ)
    g2_BZ = +elphmod.plot.toBZ(g2[iq, 0, :, :, 0, 0], **BZ)

    if comm.rank == 0:
        figure, axes = plt.subplots(1, 3)

        for i, (title, data, maximum) in enumerate([
                (r'$-\Pi$', Pi_BZ, -Pi.min()),
                (r'$-\chi^{\mathrm{b}}$', X0_BZ, -X0.min()),
                (r'$g^2$', g2_BZ, g2.max()),
                ]):

            axes[i].imshow(data)
            ckk = axes[i].contour(kx, ky, ek1_BZ, levels=[0.0], colors='k')
            ckq = axes[i].contour(kx, ky, ek2_BZ, levels=[0.0], colors='k',
                linestyles=':')

            axes[i].set_title(title)
            axes[i].axis('image')
            axes[i].axis('off')

            plot = storylines.Plot(style='APS', height=0, margin=0.1,
                xyaxes=False, background='fluctuations_%d_%d.png' % (iq, i))

            plot.width = (5 * 0.1 + 0.5 - plot.single) / 3

            plot.line(*zip(*elphmod.bravais.BZ()), thick=True)

            image = storylines.colorize(data, cmap, minimum=0, maximum=maximum)

            storylines.save(plot.background, image)

            for contour in ckk.allsegs[0]:
                plot.line(*list(zip(*contour)))

            for contour in ckq.allsegs[0]:
                plot.line(*list(zip(*contour)), color='white',
                    densely_dotted=True)

            if iq == 0:
                plot.top = 0.5
                plot.title = title.replace('Pi', 'varPi')

            if i == 0:
                plot.left = 0.5
                plot.node(-kxmax, 0, r'$\boldsymbol q = \mathrm %s$'
                    % q_label[iq], rotate=90, above=True)

            if iq == 1:
                plot.bottom = 0.5

                if i == 0:
                    plot.node(kxmax, -kymax, '0', left=True, yshift='-0.35cm',
                        inner_sep=0, outer_sep=0)

                elif i == 1:
                    dots = BZ['points']

                    colorbar = storylines.colorize([[n / (dots - 1.0)
                        for n in range(dots)]], cmap)

                    storylines.save('fluctuations.bar.png', colorbar)

                    plot.node(0, -kymax, r'\includegraphics[width=<dx=%g>cm, '
                        'height=2mm]{fluctuations.bar.png}' % (2 * kxmax),
                        yshift='-0.35cm')

                elif i == 2:
                    plot.node(-kxmax, -kymax, 'max', right=True,
                        yshift='-0.35cm', inner_sep=0, outer_sep=0)

            plot.node(-kxmax, kymax, '(%s)' % ['abc', 'def'][iq][i],
                below_right=True, inner_sep=0, outer_sep=0)

            plot.save(plot.background.replace('.png', '.pdf'))

if comm.rank == 0:
    storylines.combine('fluctuations.pdf', ['fluctuations_%d_%d' % (iq, i)
        for iq in range(len(q)) for i in range(3)], columns=3)
