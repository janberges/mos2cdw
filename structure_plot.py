#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines
import sys

xyz = sys.argv[1]

comm = elphmod.MPI.comm
info = elphmod.MPI.info

pw = elphmod.bravais.read_pwi('dft/MoS2.pwi')

a = elphmod.bravais.primitives(**pw, bohr=True)
b = np.array(elphmod.bravais.reciprocals(*a))

if comm.rank == 0:
    with open(xyz) as lines:
        next(lines)
        A = np.reshape(list(map(float, next(lines).split()[-9:])), (3, 3)).T.copy()
        N = np.round(A.dot(b.T)).astype(int)
else:
    N = np.empty((3, 3), dtype=int)

comm.Bcast(N)

info('Plotting %s x %s x %s supercell' % tuple(map(tuple, N)))

el = elphmod.el.Model('dft/MoS2_3', rydberg=True)
ph = elphmod.ph.Model('dft/MoS2.ifc', divide_mass=False, apply_asr_simple=True)
elph = elphmod.elph.Model('model/model.epmatwp', 'model/model.wigner', el, ph,
    divide_mass=False, shared_memory=True)

elph.clear()

driver = elphmod.md.Driver(elph,
    n=2 - pw['tot_charge'],
    kT=pw['degauss'],
    f=elphmod.occupations.smearing(pw['smearing']),
    supercell=N,
    unscreen=False,
)

driver.from_xyz(xyz)

r = driver.elph.ph.r
u = driver.u.reshape((-1, 3))
a = driver.elph.ph.a

tau = np.linalg.norm(r[0, :2] - r[1, :2])

sgn = -1 if np.allclose(a[0, 1:], 0.0) else +1

for na in range(driver.elph.ph.nat):
    if np.isclose(r[na] @ a[0], np.linalg.norm(r[na]) * np.linalg.norm(a[0])):
        r[na] += a[1]

    if r[na] @ a[0] < sgn * 1e-10:
        r[na] += a[0]

    if r[na] @ a[0] > 0.75 * np.linalg.norm(a[0]) ** 2 - sgn * 1e-10:
        r[na] -= a[0]

    if sgn > 0:
        phi = -np.arctan2(a[0, 1], a[0, 0])

        u[na] = elphmod.bravais.rotate(u[na], phi, two_dimensional=False)
        r[na] = elphmod.bravais.rotate(r[na], phi, two_dimensional=False)

R = r + u

driver.plot(label=False, scale=20.0, interactive=False)

if comm.rank != 0:
    raise SystemExit

plot = storylines.Plot(xyaxes=False, height=0, margin=0.2, xmin=r[:, 0].min(),
    xmax=r[:, 0].max(), ymin=r[:, 1].min(), ymax=r[:, 1].max())

bonds = storylines.bonds(R1=R[0::3, :2], R2=R[1::3, :2],
    dmin=0.9 * tau, dmax=1.1 * tau)

for bond in bonds:
    plot.line(*zip(*bond), color='lightgray')

atom = dict(mark='*', only_marks=True)

plot.line(R[1::3, 0], R[1::3, 1], color=storylines.Color(95, 151, 230), **atom)
plot.line(R[0::3, 0], R[0::3, 1], color=storylines.Color(216, 186, 141), **atom)

scale = 18.0

arrow = {'->': True}

for na in range(driver.elph.ph.nat):
    if np.linalg.norm(u[na, :2]) > 0.025:
        plot.line(
            [R[na, 0], R[na, 0] + scale * u[na, 0]],
            [R[na, 1], R[na, 1] + scale * u[na, 1]],
            line_width=max(0.3, 5 * np.linalg.norm(u[na, :2])), **arrow)

plot.save(xyz.replace('xyz', 'pdf'))
