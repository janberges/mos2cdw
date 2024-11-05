#!/usr/bin/env python3

import elphmod
import numpy as np
import matplotlib.pyplot as plt

comm = elphmod.MPI.comm
info = elphmod.MPI.info

R = np.array([list(map(float, col.split()[1:])) for col in """
ATOMIC_POSITIONS (alat)
S   0.8333541216  0.0012298005  0.1414092965
S   0.8333550411  0.0012288817 -0.1414115399
Mo  0.7456805264  0.1439265388 -0.0000017412
S   0.5842000720  0.1470286531 -0.1466648717
S   0.5842004818  0.1470316884  0.1466621901
Mo  0.5007696834  0.0087368756  0.0000005104
S   0.3333530209  0.0041619195  0.1414126690
S   0.3333549010  0.0041596247 -0.1414117206
Mo  0.2503579776  0.1479093594 -0.0000010348
S   0.0815416995  0.1481749355 -0.1398740408
S   0.0815424083  0.1481740033  0.1398723728
Mo  0.0007699264 -0.0033503930 -0.0000009556
S   0.5858861417  0.4357055716  0.1414092288
S   0.5858876834  0.4357073474 -0.1414143095
Mo  0.5061493059  0.2884287803 -0.0000023270
S   0.3345357729  0.2942423449  0.1398696405
S   0.3345341333  0.2942454125 -0.1398693133
Mo  0.2357953645  0.4357068225  0.0000024062
S   0.0841889303  0.4357081879  0.1580746619
S   0.0841888548  0.4357086321 -0.1580682850
Mo  0.0083905636  0.3044090009  0.0000015163
S  -0.1634829180  0.2896412522 -0.1398740304
S  -0.1634879649  0.2896409157  0.1398723627
Mo  0.5061502497  0.5829848724 -0.0000006310
S   0.3345362657  0.5771691714  0.1398704830
S   0.3345355319  0.5771670492 -0.1398687814
Mo  0.2503590292  0.7235035971 -0.0000006284
S   0.0815425850  0.7232387835 -0.1398731497
S   0.0815448754  0.7232402210  0.1398739185
Mo  0.0083899380  0.5670071533  0.0000025066
S  -0.1634907333  0.5817742049  0.1398722498
S  -0.1634860224  0.5817764967 -0.1398691536
Mo -0.2481252925  0.4357064156 -0.0000015068
Mo -0.2543230301  0.7274911650  0.0000014579
S  -0.4157993587  0.7243823683 -0.1466656135
S  -0.4158036465  0.7243834029  0.1466661634
""".strip().split('\n')[1:]])

a = elphmod.bravais.primitives(ibrav=4, a=1 / (2 * np.sqrt(3)))
a = elphmod.bravais.rotate(a.T, -np.pi / 6, two_dimensional=False).T
A = np.array([4 * a[0] + 2 * a[1], -2 * a[0] + 2 * a[1], a[2]])

b = np.array(elphmod.bravais.reciprocals(*a))
B = np.array(elphmod.bravais.reciprocals(*A))

plt.plot(*R.T[:2], '*')
plt.plot(*a[:2, :2].T, 's')
plt.plot(*A[:2, :2].T, 's')
plt.axis('image')
plt.show()

nq = 100

q = np.array([q1 * B[0] + q2 * B[1]
    for q1 in range(-nq, nq + 1)
    for q2 in range(-nq, nq + 1)])

maxproj = 0.5 * np.dot(b[0], b[0]) + 1e-8

q = np.array([qi for qi in q if
    maxproj >= abs(np.dot(qi, b[0])) and
    maxproj >= abs(np.dot(qi, b[1])) and
    maxproj >= abs(np.dot(qi, b[0] - b[1]))])

# q are the Gamma points of the SC BZ within the UC BZ

S = abs(np.exp(-2j * np.pi * q.dot(R.T)).sum(axis=-1)) ** 2
S /= len(R) ** 2

print(S)

#S[np.argmax(S)] = 0.0

if comm.rank == 0:
    plt.close()
    plt.scatter(q[:, 0], q[:, 1], c=S, s=68, marker='h',
        linewidth=0, aa=False)
        #, cmap='plasma')
    plt.axis('image')
    plt.axis('off')
    plt.savefig('structure_factor_2sqrt3.png', bbox_inches='tight')
