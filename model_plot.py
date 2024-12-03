#!/usr/bin/env python3

import elphmod
import matplotlib.pyplot as plt
import numpy as np

nk = 48
q = np.array([[0.0, np.pi]])

el = elphmod.el.Model('data/MoS2_3')
ph = elphmod.ph.Model('data/MoS2.ifc', apply_asr_simple=True)
elph = elphmod.elph.Model('data/MoS2_3.epmatwp', 'data/MoS2_3.wigner', el, ph)
model = elphmod.elph.Model('data/model.epmatwp', 'data/model.wigner', el, ph)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g2 = np.empty((2, 1, 1, nk, nk, el.size, el.size))

g2[0] = model.sample(q, (nk, nk), u=u[..., 1:2], squared=True)
g2[1] = elph.sample(q, (nk, nk), u=u[..., 1:2], squared=True)

figure, axes = plt.subplots(3, 6, figsize=(16, 8))

orb = ['d_{z^2}', 'd_{x^2 - y^2}', 'd_{x y}']
src = ['mod.', 'ref.']

for i in range(g2.shape[0]):
    for a in range(el.size):
        for b in range(el.size):
            image = elphmod.plot.toBZ(g2[i, 0, 0, :, :, a, b], points=100,
                outside=np.nan)

            if elphmod.MPI.comm.rank == 0:
                ax = axes[a, 2 * b + i]

                ax.set_title(r'$%s, %s$ (%s)' % (orb[a], orb[b], src[i]))
                ax.imshow(image, vmin=0, vmax=g2.max())
                ax.axis('image')
                ax.axis('off')

plt.show()
