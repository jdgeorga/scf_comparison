from ase.calculators.calculator import all_changes
from ase.io import read
from ase.calculators.singlepoint import SinglePointDFTCalculator


class QECalculator(SinglePointDFTCalculator):

    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, scf_total_path, scf_L1_path=None, scf_L2_path=None):

        self.scf_total_path = scf_total_path
        self.scf_L1_path = scf_L1_path
        self.scf_L2_path = scf_L2_path

        scf_total = read(self.scf_total_path, format='espresso-out')

        SinglePointDFTCalculator.__init__(self, scf_total,
                                          energy=scf_total.calc.results['energy'],
                                          free_energy=scf_total.calc.results['free_energy'],
                                          forces=scf_total.calc.results['forces'])

        self.results["L1_energy"] = "No L1 energy"
        self.results["L1_forces"] = "No L1 forces"
        self.results["L2_energy"] = "No L2 energy"
        self.results["L2_forces"] = "No L2 forces"

    def calculate(self, atoms, properties=None, system_changes=all_changes):

        if properties is None:
            properties = self.implemented_properties

        if self.scf_L1_path is not None:
            scf_L1 = read(self.scf_L1_path, format='espresso-out')
            self.results["L1_energy"] = scf_L1.calc.results['energy']
            self.results["L1_forces"] = scf_L1.calc.results['forces']

        if self.scf_L2_path is not None:
            scf_L2 = read(self.scf_L2_path, format='espresso-out')
            self.results["L2_energy"] = scf_L2.calc.results['energy']
            self.results["L2_forces"] = scf_L2.calc.results['forces']

        if self.scf_L1_path is not None and self.scf_L2_path is not None:
            IL_energy = self.results["energy"] - self.results["L1_energy"] - self.results["L2_energy"]

            L1_cond = atoms.positions[:, 2] < atoms.positions[:, 2].mean()
            L2_cond = atoms.positions[:, 2] >= atoms.positions[:, 2].mean()
            IL_forces = self.results["forces"].copy()
            IL_forces[L1_cond] -= self.results['L1_forces']
            IL_forces[L2_cond] -= self.results['L2_forces']

            self.results['IL_energy'] = IL_energy
            self.results['IL_forces'] = IL_forces
