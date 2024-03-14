from ase.io import read, write
from pathlib import Path
from moirecompare.calculators import AllegroCalculator
from moirecompare.calculators import MonolayerLammpsCalculator, BilayerLammpsCalculator,InterlayerLammpsCalculator, BilayerCalculator
from ase.calculators.lammpslib import LAMMPSlib
from moirecompare.calculators import QECalculator
from ase.optimize import FIRE
import numpy as np



# BILAYER
xyz_file_path = "BP_reference_0-24_70.53.xyz"
# xyz_file_path = "structures/01-3.21_deg-10208_atoms.xyz"

at = read(xyz_file_path)

# Setting up structure
IL_sep = 2.0
bottom_layer_disp = -IL_sep/2 - at.positions[at.arrays['atom_types'] < 4, 2].max()
top_layer_disp = IL_sep/2 - at.positions[at.arrays['atom_types'] >= 4, 2].min()
at.positions[at.arrays['atom_types'] < 4, 2] += bottom_layer_disp
at.positions[at.arrays['atom_types'] >= 4, 2] += top_layer_disp
# write("structures/superclose_10208.cif", at, format="cif")

at_1 = at[at.arrays['atom_types'] < 4]
at_2 = at[at.arrays['atom_types'] >= 4]

calc_1 = MonolayerLammpsCalculator(at_1,
                                   chemical_symbols= ['P','P','P', 'P'],
                                   system_type='BP')
calc_2 = MonolayerLammpsCalculator(at_2,
                                   chemical_symbols= ['P','P','P', 'P'],
                                   system_type='BP')
# calc_IL = InterlayerLammpsCalculator(at,
#                                      chemical_symbols = [['P','P','P', 'P'],
#                                                          ['P','P','P', 'P']],
#                                      system_type='BP')

# calc_BL = BilayerLammpsCalculator(at, chemical_symbols= [['P','P','P', 'P'],
#                                                       ['P','P','P', 'P']],
#                                                       system_type='BP')

calc_ALLEGRO = AllegroCalculator(4, 
                         ["P1L1", "P2L1", "P3L1", "P4L1", "P1L2", "P2L2", "P3L2", "P4L2"],
                         intralayer_symbol_type={"P": 0},
                         device='cpu')

calc_ALLEGRO.setup_models(
    2,  # Prolly not important

    [Path("./intralayer_bp.pth"), Path("./intralayer_bp.pth")],
    [Path("./interlayer_bp.pth")],
    IL_factor=1.0,
    L1_factor=0.0,
    L2_factor=0.0,
)

calc = BilayerCalculator(calc_1, calc_2, calc_ALLEGRO,
                         [['P', 'P', 'P', 'P'],
                          ['P', 'P', 'P', 'P']])
at.calc = calc
at.calc.calculate(at)

il_dist = at.positions[at.arrays['atom_types'] < 4, 2].max() - at.positions[at.arrays['atom_types'] >= 4, 2].min()
print(at.positions[at.arrays['atom_types'] < 4, 2].max(),il_dist)

print(f"Unrelaxed: Total_energy {at.calc.results['energy']:.3f} eV, ",
      f"L1_energy {at.calc.results['L1_energy']:.3f} eV, ",
      f"L2_energy {at.calc.results['L2_energy']:.3f} eV, ",
      f"Interlayer_energy {at.calc.results['IL_energy']:.3f} eV")

dyn = FIRE(at,maxstep=0.1)
dyn.run(fmax=3e-3)


il_dist = at.positions[at.arrays['atom_types'] < 4, 2].max() - at.positions[at.arrays['atom_types'] >= 4, 2].min()
print(at.positions[at.arrays['atom_types'] < 4, 2].max(), il_dist)
print(f"Relaxed: Total_energy {at.calc.results['energy']:.3f} eV, ",
        f"L1_energy {at.calc.results['L1_energy']:.3f} eV, ",
        f"L2_energy {at.calc.results['L2_energy']:.3f} eV, ",
        f"Interlayer_energy {at.calc.results['IL_energy']:.3f} eV")

# print("Writing relaxed structure to relaxed.cif")
# write("relaxed_structures/test.cif", at, format="cif")
