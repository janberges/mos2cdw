# Charge-density waves in molybdenum disulfide

We use the electron-phonon coupling data from Phys. Rev. X 13, 041009 (2023),
see Fig. 9, which is located in the directory `dft`. To reduce the computational
cost, we extract a three-band model (`dz2`, `dx2-y2`, `dxy`) from this five-band
model (which also contains `dxz` and `dyz` orbitals):

    python3 523.py

To further reduce the cost, we map the coupling to a nearest-neighbor model:

    cd model
    python3 optimize.py

Now we are ready to set up different supercells:

    python3 setup_small.py
    python3 setup_large.py

Then we can perform structural relaxations on these supercells:

    python3 relax_small.py
    python3 relax_large.py

Finally, we calculate the doping-dependent superconducting critical temperature:

    mpirun python3 phases.py
    python3 phases_plot.py
