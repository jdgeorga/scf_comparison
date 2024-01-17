from ase.io import read
from pathlib import Path
from moirecompare.calculators import AllegroCalculator

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