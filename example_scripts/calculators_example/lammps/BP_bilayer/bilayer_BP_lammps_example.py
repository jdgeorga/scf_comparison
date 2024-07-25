from ase.io import read, write
from pathlib import Path
from moirecompare.calculators import AllegroCalculator
from moirecompare.calculators import MonolayerLammpsCalculator, BilayerLammpsCalculator
from ase.calculators.lammpslib import LAMMPSlib
from moirecompare.calculators import QECalculator
from ase.optimize import FIRE


xyz_file_path = "BP_reference_0-24_70.53.xyz"

at = read(xyz_file_path)


# Setting up structure
IL_sep = 2.5
bottom_layer_disp = -IL_sep/2 - at.positions[at.arrays['atom_types'] < 4, 2].max()
top_layer_disp = IL_sep/2 - at.positions[at.arrays['atom_types'] >= 4, 2].min()
at.positions[at.arrays['atom_types'] < 4, 2] += bottom_layer_disp
at.positions[at.arrays['atom_types'] >= 4, 2] += top_layer_disp


# calc = MonolayerLammpsCalculator(at, chemical_symbols= ['P','P','P', 'P'],
#                                                       system_type='BP')
calc = BilayerLammpsCalculator(at, chemical_symbols= [['P','P','P', 'P'],
                                                      ['P','P','P', 'P']],
                                                      system_type='BP')

at.calc = calc
at.calc.calculate(at)
print(f"Unrelaxed: Total_energy {at.calc.results['energy']:.3f} eV, ",
      f"L1_energy {at.calc.results['L1_energy']:.3f} eV, ",
      f"L2_energy {at.calc.results['L2_energy']:.3f} eV, ",
      f"Interlayer_energy {at.calc.results['IL_energy']:.3f} eV")

at.calc.minimize(at)

print(f"Relaxed: Total_energy {at.calc.results['energy']:.3f} eV, ",
        f"L1_energy {at.calc.results['L1_energy']:.3f} eV, ",
        f"L2_energy {at.calc.results['L2_energy']:.3f} eV, ",
        f"Interlayer_energy {at.calc.results['IL_energy']:.3f} eV")

print("Writing relaxed structure to relaxed.cif")
write("relaxed.cif", at, format="cif")