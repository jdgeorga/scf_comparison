from ase.calculators.lammpslib import LAMMPSlib

from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)

from ase.data import atomic_numbers, atomic_masses
from moirecompare.utils import rotate_to_x_axis


class BilayerLammpsCalculator(LAMMPSlib):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, atoms, chemical_symbols, system_type='TMD'):

        self.system_type = system_type
        self.original_atom_types = chemical_symbols
        self.original_masses = [atomic_masses[m] for m in [atomic_numbers[c] for c in chemical_symbols]]
        print(self.original_atom_types, self.original_masses)

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style hybrid/overlay sw sw kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 lj/cut 10.0",
                f"pair_coeff * * sw 1 tmd.sw {self.original_atom_types[0]} {self.original_atom_types[1]} {self.original_atom_types[2]} NULL NULL NULL",
                f"pair_coeff * * sw 2 tmd.sw NULL NULL NULL {self.original_atom_types[3]} {self.original_atom_types[4]} {self.original_atom_types[5]}",
                f"pair_coeff 1 6 kolmogorov/crespi/z 1 WS.KC  {self.original_atom_types[0]} NULL NULL NULL NULL  {self.original_atom_types[5]}",
                f"pair_coeff 2 4 kolmogorov/crespi/z 2 WS.KC NULL  {self.original_atom_types[1]} NULL {self.original_atom_types[3]} NULL NULL",
                f"pair_coeff 2 6 kolmogorov/crespi/z 3 WS.KC NULL  {self.original_atom_types[1]} NULL NULL NULL  {self.original_atom_types[5]}",
                f"pair_coeff 1 4 kolmogorov/crespi/z 4 WS.KC  {self.original_atom_types[0]} NULL NULL {self.original_atom_types[3]} NULL NULL",
                "pair_coeff * * lj/cut 0.0 3.0",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            # Define fixed atom types and masses for the simulation.

            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
            fixed_atom_type_masses = {'H': self.original_masses[0],
                                      'He': self.original_masses[1],
                                      'Li': self.original_masses[2],
                                      'Be': self.original_masses[3],
                                      'B': self.original_masses[4],
                                      'C': self.original_masses[5]} 
            # Set up the LAMMPS calculator with the specified commands.

            LAMMPSlib.__init__(self,
                      lmpcmds=cmds,
                      atom_types=fixed_atom_types,
                      atom_type_masses=fixed_atom_type_masses,
                      keep_alive = True)
        
    def calculate(self, 
                  atoms, 
                  properties=None,
                  system_changes=all_changes):

        atoms = rotate_to_x_axis(atoms)
        if properties is None:
            properties = self.implemented_properties

        # self.system_type = system_type
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.propagate(atoms, properties, system_changes, 0)
            self.tmd_calculate_intralayer(atoms)
            self.tmd_calculate_interlayer(atoms)
        
    def tmd_calculate_intralayer(self, atoms):
        print(atoms)
        L1_atom_types = self.original_atom_types[:3]
        L1_atom_masses = self.original_masses[:3]
        atom_L1 = atoms[
            atoms.positions[:, 2] < atoms.positions[:, 2].mean()
            ]
        atom_L1.numbers = atom_L1.arrays["atom_types"] + 1

        cmds = [
            "pair_style hybrid/overlay sw lj/cut 10.0",
            f"pair_coeff * * sw tmd.sw {L1_atom_types[0]} {L1_atom_types[1]} {L1_atom_types[2]}",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]
        print("L1_cmds", cmds)


        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': L1_atom_masses[0],
                            'He': L1_atom_masses[1],
                            'Li': L1_atom_masses[2]}
        # Set up and run the LAMMPS simulation for the first layer.
        lammps_L1 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses,
                              log_file='log_L1.txt',
                              keep_alive = False)
        atom_L1.calc = lammps_L1
        L1_energy = atom_L1.get_potential_energy()
        L1_forces = atom_L1.get_forces()
        print(atom_L1.cell)

        # Repeat similar calculations for the second layer.
        L2_atom_types = self.original_atom_types[3:]
        L2_atom_masses = self.original_masses[3:]

        atom_L2 = atoms[
            atoms.positions[:, 2] >= atoms.positions[:, 2].mean()
            ]
        atom_L2.numbers = atom_L2.arrays["atom_types"] - 3 + 1


        cmds = [
            "pair_style hybrid/overlay sw lj/cut 10.0",
            f"pair_coeff * * sw tmd.sw {L2_atom_types[0]} {L2_atom_types[1]} {L2_atom_types[2]}",
            "pair_coeff * * lj/cut 0.0 3.0",
            "neighbor        2.0 bin",
            "neigh_modify every 1 delay 0 check yes"]
        print("L2_cmds", cmds)

        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': L2_atom_masses[0],
                            'He': L2_atom_masses[1],
                            'Li': L2_atom_masses[2]}
        lammps_L2 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses,
                              log_file='log_L2.txt')
        atom_L2.calc = lammps_L2
        
        L2_energy = atom_L2.get_potential_energy()
        L2_forces = atom_L2.get_forces()

        self.results["L1_energy"] = L1_energy
        self.results["L1_forces"] = L1_forces
        self.results["L2_energy"] = L2_energy
        self.results["L2_forces"] = L2_forces

    def tmd_calculate_interlayer(self, atoms):

        IL_energy = self.results["energy"] - self.results["L1_energy"] - self.results["L2_energy"]

        L1_cond = atoms.positions[:, 2] < atoms.positions[:, 2].mean()
        L2_cond = atoms.positions[:, 2] >= atoms.positions[:, 2].mean()
        IL_forces = self.results["forces"].copy()
        IL_forces[L1_cond] -= self.results['L1_forces']
        IL_forces[L2_cond] -= self.results['L2_forces']

        self.results['IL_energy'] = IL_energy
        self.results['IL_forces'] = IL_forces

    def minimize(self, atoms, method="fire"):
        min_cmds = [f"min_style       {method}",
                    f"minimize        0.0 1.0e-8 10000000 10000000",
                    "write_data      lammps.dat_min"]

        self.set(amendments=min_cmds)
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions"],
                       n_steps=1)


class MonolayerLammpsCalculator(LAMMPSlib):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, atoms, chemical_symbols, system_type='TMD'):

        self.system_type = system_type
        self.original_atom_types = atoms.get_chemical_symbols()
        self.original_masses = [atomic_masses[m] for m in [atomic_numbers[c] for c in chemical_symbols]]
        # print(self.original_atom_types, self.original_masses)
        print(chemical_symbols)

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style sw",
                f"pair_coeff * * sw tmd.sw {chemical_symbols[0]} {chemical_symbols[1]} {chemical_symbols[2]}",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            # Define fixed atom types and masses for the simulation.

            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3}
            fixed_atom_type_masses = {'H': self.original_masses[0],
                                      'He': self.original_masses[1],
                                      'Li': self.original_masses[2]}
            # Set up the LAMMPS calculator with the specified commands.

            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                               log_file='log.txt',
                               keep_alive=True)
            
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system.
            cmds = [
                "pair_style sw",
                # f"pair_coeff * * sw bp.sw {chemical_symbols[0]} {chemical_symbols[1]} {chemical_symbols[2]} {chemical_symbols[3]}",
                f"pair_coeff * * bp.sw T T B B",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation.
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4}
            fixed_atom_type_masses = {'H': self.original_masses[0],
                                        'He': self.original_masses[1],
                                        'Li': self.original_masses[2],
                                        'Be': self.original_masses[3]}
            
            # Set up the LAMMPS calculator with the specified commands.
            LAMMPSlib.__init__(self,
                                 lmpcmds=cmds,
                                 atom_types=fixed_atom_types,
                                 atom_type_masses=fixed_atom_type_masses,
                                 log_file='log.txt',
                                 keep_alive=True)
        
    def calculate(self, 
                  atoms, 
                  properties=None,
                  system_changes=all_changes):

        # atoms = rotate_to_x_axis(atoms)
        if properties is None:
            properties = self.implemented_properties

        # self.system_type = system_type
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.propagate(atoms, properties, system_changes, 0)

        elif self.system_type == 'BP':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            print(atoms)
            self.propagate(atoms, properties, system_changes, 0)
        

    def minimize(self, atoms, method="fire"):
        min_cmds = [f"min_style       {method}",
                    f"minimize        0.0 1.0e-4 10000000 10000000",
                    "write_data      lammps.dat_min"]

        self.set(amendments=min_cmds)
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions",'pbc'],
                       n_steps=1)
        
    def clean_atoms(self, atoms):
        atoms.set_chemical_symbols(self.original_atom_types)
        if self.started:
            self.lmp.close()
            self.started = False
            self.initialized = False
            self.lmp = None

        
    # def reset_atoms(self):
        
        
    #     self.atoms.numbers = atoms.arrays["atom_types"] + 1
    #     self.atoms.set_calculator(self)
    #     self.atoms.calc = self