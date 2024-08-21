#!/usr/bin/env python3

import elphmod

energy0 = None
energy = None
doping = None
init = None

for cell, cells in ('18', 18 ** 2), ('12sqrt3', 12 ** 2 * 3), ('27', 27 ** 3), ('18sqrt3', 18 ** 2 * 3):
    with open('relax_%s.out' % cell) as lines:
        while True:
            for line in lines:
                if line.startswith('Free energy'):
                    energy = float(line.split()[2])
                elif line.startswith('Total force'):
                    continue
                else:
                    if energy is not None:
                        if doping == 0.25:
                            energy0 = energy

                        with open('relax_%s_%s_%.2f.txt' % (cell, init, doping), 'w') as data:
                            data.write('%.2f' % (1e3 * elphmod.misc.Ry * (energy - energy0) / cells))

                        energy = None

                    break
            else:
                if energy is not None:
                    if doping == 0.25:
                        energy0 = energy

                    with open('relax_%s_%s_%.2f.txt' % (cell, init, doping), 'w') as data:
                        data.write('%.2f' % (1e3 * elphmod.misc.Ry * (energy - energy0) / cells))

                    energy = None

                break

            if line.startswith('Doping'):
                doping = float(line.split()[-1])

            elif line.startswith('Polaron') or line.startswith('CDW'):
                init = line.strip().lower()
