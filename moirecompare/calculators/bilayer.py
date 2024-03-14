from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase import Atoms
import numpy as np


class BilayerCalculator(Calculator):
    implemented_properties = ['energy', 'energies', 'forces']  # Define the properties this calculator can compute
    
    def __init__(self,
                 calc_L1,
                 calc_L2,
                 calc_IL,
                 layer_symbols,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calc_L1 = calc_L1
        self.calc_L2 = calc_L2
        self.calc_IL = calc_IL
        self.layer_symbols = layer_symbols


    def calculate(self,
                  atoms: Atoms,
                  properties=None,
                  system_changes=all_changes):
        
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        self.calculate_intralayer(atoms, layer=1)
        self.calculate_intralayer(atoms, layer=2)
        self.calculate_interlayer(atoms)
        
        L1_cond = atoms.arrays["atom_types"] < len(self.layer_symbols[0])
        L2_cond = atoms.arrays["atom_types"] >= len(self.layer_symbols[0])

        self.results["energy"] = self.results["L1_energy"] + self.results["L2_energy"] + self.results["IL_energy"]
        self.results["forces"] = self.results["IL_forces"]
        self.results["forces"][L1_cond] += self.results["L1_forces"]
        self.results["forces"][L2_cond] += self.results["L2_forces"]
        self.results['free_energy'] = self.results['energy']

     
    def calculate_intralayer(self, atoms, layer):
        if layer == 1:
            calc = self.calc_L1
            atoms_L = atoms.copy()[atoms.arrays["atom_types"] < len(self.layer_symbols[0])]
        elif layer == 2:
            calc = self.calc_L2
            atoms_L = atoms.copy()[atoms.arrays["atom_types"] >= len(self.layer_symbols[0])]
        else:
            raise ValueError("layer must be 1 or 2")
        
        atoms_L.calc = calc
        atoms_L.calc.calculate(atoms_L)
        self.results[f"L{layer}_energy"] = atoms_L.get_potential_energy()
        self.results[f"L{layer}_forces"] = atoms_L.get_forces()

    def calculate_interlayer(self, atoms):
        tmp_atoms = atoms.copy()[:]
        calc = self.calc_IL
        tmp_atoms.calc = calc
        tmp_atoms.calc.calculate(tmp_atoms)
        self.results["IL_energy"] = tmp_atoms.get_potential_energy()
        self.results["IL_forces"] = tmp_atoms.get_forces()