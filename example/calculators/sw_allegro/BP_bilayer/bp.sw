# by Jin-Wu Jiang, jwjiang5918@hotmail.com, 06/04/15/Mon

# The Stillinger-Weber (SW) parameters for single-layer black phosphorus (SLBP).
# (0). Include this potential in LAMMPS input script as follows,
#      pair_style      sw
#      pair_coeff      * * bp.sw T B
# (1). SW parameters in GULP are derived analytically from the valence force field model.
# (2). Atoms in SLBP are divided into the top (T) group and the bottom (B) group.

# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3, 
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q, tol

# intra-group SW2 and SW3
T T T 1.000   0.809   3.449  35.701   1.000  -0.111   3.626  33.371  4  0 0.0
B B B 1.000   0.809   3.449  35.701   1.000  -0.111   3.626  33.371  4  0 0.0

# inter-group SW2
T B B 1.000   0.809   3.449   0.000   1.000  -0.111   3.626  33.371  4  0 0.0
B T T 1.000   0.809   3.449   0.000   1.000  -0.111   3.626  33.371  4  0 0.0

# inter-group SW3
T T B 1.000   0.809   3.449  32.006   1.000  -0.210   0.000  33.371  4  0 0.0
T B T 1.000   0.809   3.449  32.006   1.000  -0.210   0.000  33.371  4  0 0.0

# inter-group SW3
B B T 1.000   0.809   3.449  32.006   1.000  -0.210   0.000  33.371  4  0 0.0
B T B 1.000   0.809   3.449  32.006   1.000  -0.210   0.000  33.371  4  0 0.0
