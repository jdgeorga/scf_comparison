from ase.io import read, write

from moirecompare.calculators import (
                                      NLayerCalculator,
                                      AllegroCalculator)
import numpy as np

from pathlib import Path
from ase.io import read

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from ase.optimize import FIRE


xyz_file_path = "11-70.53_deg-24_atoms.xyz"

atoms = read(xyz_file_path)

# Setting up structure
IL_sep = 2.0
bottom_layer_disp = -IL_sep/2 - atoms.positions[atoms.arrays['atom_types'] < 4, 2].max()
top_layer_disp = IL_sep/2 - atoms.positions[atoms.arrays['atom_types'] >= 4, 2].min()
atoms.positions[atoms.arrays['atom_types'] < 4, 2] += bottom_layer_disp
atoms.positions[atoms.arrays['atom_types'] >= 4, 2] += top_layer_disp
# write("structures/superclose_10208.cif", at, format="cif")

at_1 = atoms[atoms.arrays['atom_types'] < 4]
at_2 = atoms[atoms.arrays['atom_types'] >= 4]

# add double layer 1 on top of layer 2

new_layer = atoms[atoms.arrays["atom_types"] < 4].copy()[:]

new_layer_disp = atoms.positions[atoms.arrays["atom_types"] >= 4, 2].max()
new_layer_disp += IL_sep
new_layer_disp -= atoms.positions[atoms.arrays["atom_types"] < 4, 2].min()
new_layer.positions[:, 2] += new_layer_disp
new_layer.arrays["atom_types"] += 8

atoms += new_layer
write("trilayer_unrelaxed.cif", atoms, format="cif")

print(atoms.arrays['atom_types'])

at_1 = atoms[atoms.arrays['atom_types'] < 4]
at_2 = atoms[np.logical_and(atoms.arrays['atom_types'] >= 4,
                         atoms.arrays['atom_types'] < 8)]
at_3 = atoms[atoms.arrays['atom_types'] >= 8]

layer_symbols = [['P','P','P', 'P'],
                 ['P','P','P', 'P'],
                 ['P','P','P', 'P']]

interlayer_model = "./interlayer_bp.pth"
intralayer_model = "./intralayer_bp.pth"

print(layer_symbols[1:3])

intra_calc_1 = AllegroCalculator(at_1,
                               layer_symbols[0],
                                model_file=intralayer_model)
intra_calc_2 = AllegroCalculator(at_2,
                               layer_symbols[1],
                                model_file=intralayer_model)

intra_calc_3 = AllegroCalculator(at_3,
                                 layer_symbols[2],
                                  model_file=intralayer_model)


calc_IL_1 = AllegroCalculator(atoms[atoms.arrays['atom_types'] < 8],
                                       layer_symbols=layer_symbols[0:2],
                                       model_file=interlayer_model)

calc_IL_2 = AllegroCalculator(atoms[atoms.arrays['atom_types'] >= 4],
                                       layer_symbols=layer_symbols[1:3],
                                       model_file=interlayer_model)


nlayer_calc = NLayerCalculator(atoms,
                               [intra_calc_1, intra_calc_2, intra_calc_3], 
                               [calc_IL_1, calc_IL_2],
                               layer_symbols=layer_symbols)

# atoms = atoms[atoms.arrays['atom_types'] < 8]
print(layer_symbols[0:2])
atoms.calc = nlayer_calc
atoms.calc.calculate(atoms)


print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
      f"layer_energy {atoms.calc.results['layer_energy']}")

dyn = FIRE(atoms,trajectory = 'tri_BP_relax.traj')
dyn.run(fmax=5e-3)

print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
      f"layer_energy {atoms.calc.results['layer_energy']}")

# print("Writing relaxed structure to relaxed.cif")
write("test.cif", atoms, format="cif")

