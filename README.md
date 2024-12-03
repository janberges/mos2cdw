# Supercell model calculations for doped MoS₂

This directory contains the data and scripts to calculate lattice distortions
and superconducting critical temperatures for electron-doped MoS₂ monolayer on
large supercells, as shown by Girotto, Berges, Poncé, and Novko (2024).

Reproducing the calculations requires the installation of the Python packages
elphmod and StoryLines (e.g., in a virtual environment):

    python3 -m pip install elphmod==0.29 storylines==0.15

We use the electron-phonon coupling data from Phys. Rev. X 13, 041009 (2023),
see Fig. 9, which is located in the folder `data`. To reduce the computational
cost, we extract a three-band model (Mo `dz2`, `dx2-y2`, and `dxy` orbitals)
from this five-band model (which also contains Mo `dxz` and `dyz` orbitals):

    python3 523.py

To further reduce the cost, we map the coupling to a nearest-neighbor model, as
described in Appendix B of Phys. Rev B 101, 155107 (2020):

    python3 model_optimize.py
    python3 model_plot.py

Now we are ready to relax the structure on different supercells, employing the
“model III” presented in SciPost Phys. 16, 046 (2024):

    python3 relax_2x2.py
    python3 relax_8x8.py

Finally, we calculate the doping-dependent superconducting critical temperature
as a function of doping, which yields the well-known dome structure (differences
between the calculated points and the reference lines are due to insufficient k-
and q-point densities):

    mpirun phases_2x2.py
    python3 phases_2x2_plot.py

The calculations on the largest supercell are best performed on a supercomputer.
The results are plotted for Fig. 6 and Supplementary Figure 7 of the paper:

    sbatch phases_18sqrt3.sh
    python3 phases_18sqrt3_plot.py

A presentation of the relaxed structures for all dopings can also be created:

    python3 structure_plot.py phases_18sqrt3/*.xyz
    python3 phases_18sqrt3_overview.py
    pdflatex phases_18sqrt3_overview.tex
