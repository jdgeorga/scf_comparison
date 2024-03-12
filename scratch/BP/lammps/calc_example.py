from ase.io import read
from pathlib import Path
from moirecompare.calculators import AllegroCalculator
from moirecompare.calculators import MonolayerLammpsCalculator
from ase.calculators.lammpslib import LAMMPSlib
from moirecompare.calculators import QECalculator
from ase.optimize import FIRE


xyz_file_path = "BP_reference_0-24_70.53.xyz"

at = read(xyz_file_path)

# calc = AllegroCalculator(4, 
#                          ["MoL1", "SL1", "SeL1", "MoL2", "SL2", "SeL2"],
#                          intralayer_symbol_type={"Mo": 0, "S": 1},
#                          device='cpu')
# calc.setup_models(
#     2,
#     [Path("./intralayer_beefy.pth"), Path("./intralayer_beefy.pth")],
#     [Path("./interlayer_beefy_truncated.pth")],
# )
# at = read("./MoS2-Bilayer.xyz")
# at.calc = calc
# at.calc.calculate(at)
# print(at.calc.results)

# dyn = FIRE(at)
# dyn.run(fmax=0.01)
# print(at.positions)


# # Example usage of LammpsCalculator

# Create an instance of LammpsCalculator
at = at[at.arrays['atom_types'] < 4]
calc = MonolayerLammpsCalculator(at, chemical_symbols= ['P','P','P', 'P'],
                                                      system_type='BP')
print(at.positions)

at.calc = calc
at.calc.calculate(at)
at.calc.minimize(at)
print(at.positions)
print("LAMMPS", at.calc.results)
