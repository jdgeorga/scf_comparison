from ase.io import read
from pathlib import Path
from moirecompare.calculators import AllegroCalculator
from moirecompare.calculators import BilayerLammpsCalculator
from ase.calculators.lammpslib import LAMMPSlib
from moirecompare.calculators import QECalculator


xyz_file_path = "MoS2-Bilayer.xyz"

at = read(xyz_file_path)

calc = AllegroCalculator(4, 
                         ["MoL1", "SL1", "SeL1", "MoL2", "SL2", "SeL2"],
                         device='cpu')
calc.setup_models(
    2,
    [Path("./intralayer_beefy.pth"), Path("./intralayer_beefy.pth")],
    [Path("./interlayer_beefy_truncated.pth")],
)
at = read("./MoS2-Bilayer.xyz")
at.calc = calc
at.calc.calculate(at)
print(at.calc.results)

# Example usage of LammpsCalculator

# Create an instance of LammpsCalculator
calc = BilayerLammpsCalculator(at, system_type='TMD')
at.calc = calc
at.calc.calculate(at)
print("LAMMPS", at.calc.results)

at.calc = QECalculator("scf_total.out", "scf_L1.out", "scf_L2.out")
at.calc.calculate(at)
print("QE", at.calc.results)