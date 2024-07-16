from ase.calculators.lammpslib import LAMMPSlib

from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)

from ase.data import atomic_numbers, atomic_masses
# from moirecompare.utils import rotate_to_x_axis
import numpy as np


class MonolayerLammpsCalculator(LAMMPSlib):
    implemented_properties = ["energy", "energies", "forces"]

    def __init__(self, atoms, layer_symbols, system_type='TMD', intra_potential='tmd.sw'):

        self.system_type = system_type
        self.layer_symbols = layer_symbols
        self.original_masses = [atomic_masses[m] for m in [atomic_numbers[c] for c in self.layer_symbols]]
        self.original_chemical_symbols = atoms.get_chemical_symbols()
        self.atom_types = atoms.arrays['atom_types']
        self.intra_potential = intra_potential

        # Get unique types and their indices in the sorted order
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)
        
        # Create a new array with numbers replaced by their size order
        self.relative_layer_types = inverse

        # checks if atom_types match the layer_symbols
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("More/fewer atom types in Monolayer than layer_symbols provided.")

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style sw/mod",
                f"pair_coeff * * {self.intra_potential} {self.layer_symbols[0]} {self.layer_symbols[1]} {self.layer_symbols[2]}",
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
                            #    log_file='log.txt',
                               keep_alive=True)
            
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system.
            cmds = [
                "pair_style sw/mod maxdelcs 0.25 0.35",
                f"pair_coeff * * {self.intra_potential} T T B B",
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
                                #  log_file='log_ML.txt',
                                 keep_alive=True)
    
    def calculate(self, 
                  atoms, 
                  properties=None,
                  system_changes=all_changes):

        # atoms = rotate_to_x_axis(atoms)
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # self.system_type = system_type
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'BP'")

        if self.system_type == 'TMD':
            atoms.numbers = self.relative_layer_types + 1
            self.propagate(atoms, properties, system_changes, 0)
            atoms.set_chemical_symbols(self.original_chemical_symbols)

        elif self.system_type == 'BP':
            atoms.numbers = self.relative_layer_types + 1
            self.propagate(atoms, properties, system_changes, 0)
            atoms.set_chemical_symbols(self.original_chemical_symbols)

    def minimize(self, atoms, method="fire"):
        min_cmds = [f"min_style       {method}",
                    f"minimize        1.0e-6 1.0e-8 10000 10000",
                    "write_data      lammps.dat_min"]

        self.set(amendments=min_cmds)
        atoms.numbers = atoms.arrays["atom_types"] + 1
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions",'pbc'],
                       n_steps=1)
        atoms.set_chemical_symbols(self.original_chemical_symbols)
        
    def clean_atoms(self, atoms):
        atoms.set_chemical_symbols(self.original_chemical_symbols)
        if self.started:
            self.lmp.close()
            self.started = False
            self.initialized = False
            self.lmp = None
         
class InterlayerLammpsCalculator(LAMMPSlib):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, atoms, layer_symbols, system_type='TMD'):

        self.system_type = system_type
        self.layer_symbols = layer_symbols
        self.original_chemical_symbols = atoms.get_chemical_symbols()
        self.original_masses = [[atomic_masses[atomic_numbers[n]] for n in t] for t in self.layer_symbols]
        self.atom_types = atoms.arrays['atom_types']

        # Get unique types and their indices in the sorted order
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)
        
        # Create a new array with numbers replaced by their size order
        self.relative_layer_types = inverse

        # checks if atom_types match the layer_symbols
        if len(unique_types) != len([a for b in layer_symbols for a in b]):
            raise ValueError("More/fewer atom types in Monolayer than layer_symbols provided.")

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style hybrid/overlay kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 lj/cut 10.0",
                f"pair_coeff 1 6 kolmogorov/crespi/z 1 WS.KC  {self.layer_symbols[0][0]} NULL NULL NULL NULL  {self.layer_symbols[1][2]}",
                f"pair_coeff 2 4 kolmogorov/crespi/z 2 WS.KC NULL  {self.layer_symbols[0][1]} NULL {self.layer_symbols[1][0]} NULL NULL",
                f"pair_coeff 2 6 kolmogorov/crespi/z 3 WS.KC NULL  {self.layer_symbols[0][1]} NULL NULL NULL  {self.layer_symbols[1][2]}",
                f"pair_coeff 1 4 kolmogorov/crespi/z 4 WS.KC  {self.layer_symbols[0][0]} NULL NULL {self.layer_symbols[1][0]} NULL NULL",
                "pair_coeff * * lj/cut 0.0 3.0",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            # Define fixed atom types and masses for the simulation.

            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[1][0],
                                      'B': self.original_masses[1][1],
                                      'C': self.original_masses[1][2]} 
            # Set up the LAMMPS calculator with the specified commands.

            LAMMPSlib.__init__(self,
                      lmpcmds=cmds,
                      atom_types=fixed_atom_types,
                      atom_type_masses=fixed_atom_type_masses,
                      keep_alive = True)
 
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system.
            cmds = [
                "pair_style lj/cut 21.0",
                "pair_coeff * * 0.0 3.695",
                "pair_coeff 1*4 5*8 0.0103 3.405",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation.
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[0][3],
                                      'B': self.original_masses[1][0],
                                      'C': self.original_masses[1][1],
                                      'N': self.original_masses[1][2],
                                      'O': self.original_masses[1][3]}
            
            # Set up the LAMMPS calculator with the specified commands.
            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                            #    log_file='log_IL.txt',
                               keep_alive=True)
        
    def calculate(self,
                  atoms,
                  properties=None,
                  system_changes=all_changes):

        # atoms = rotate_to_x_axis(atoms)
        if properties is None:
            properties = self.implemented_properties
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        atoms.numbers = self.relative_layer_types + 1
        self.propagate(atoms, properties, system_changes, 0)
        atoms.set_chemical_symbols(self.original_chemical_symbols)



class BilayerLammpsCalculator_old(LAMMPSlib):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self, atoms, chemical_symbols, system_type='TMD'):
        self.system_type = system_type
        self.original_atom_types = atoms.get_chemical_symbols()
        self.chemical_symbols = chemical_symbols

        self.original_masses = [[atomic_masses[atomic_numbers[n]] for n in t] for t in self.chemical_symbols]

        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style hybrid/overlay sw sw kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 lj/cut 10.0",
                f"pair_coeff * * sw 1 tmd.sw {self.chemical_symbols[0][0]} {self.chemical_symbols[0][1]} {self.chemical_symbols[0][2]} NULL NULL NULL",
                f"pair_coeff * * sw 2 tmd.sw NULL NULL NULL {self.chemical_symbols[1][0]} {self.chemical_symbols[1][1]} {self.chemical_symbols[1][2]}",
                f"pair_coeff 1 6 kolmogorov/crespi/z 1 WS.KC  {self.chemical_symbols[0][0]} NULL NULL NULL NULL  {self.chemical_symbols[1][2]}",
                f"pair_coeff 2 4 kolmogorov/crespi/z 2 WS.KC NULL  {self.chemical_symbols[0][1]} NULL {self.chemical_symbols[1][0]} NULL NULL",
                f"pair_coeff 2 6 kolmogorov/crespi/z 3 WS.KC NULL  {self.chemical_symbols[0][1]} NULL NULL NULL  {self.chemical_symbols[1][2]}",
                f"pair_coeff 1 4 kolmogorov/crespi/z 4 WS.KC  {self.chemical_symbols[0][0]} NULL NULL {self.chemical_symbols[1][0]} NULL NULL",
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
 
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system.
            cmds = [
                "pair_style hybrid/overlay sw sw lj/cut 21.0",
                # f"pair_coeff * * sw bp.sw {chemical_symbols[0]} {chemical_symbols[1]} {chemical_symbols[2]} {chemical_symbols[3]}",
                f"pair_coeff * * sw 1 bp.sw T T B B NULL NULL NULL NULL",
                f"pair_coeff * * sw 2 bp.sw NULL NULL NULL NULL T T B B",
                "pair_coeff * * lj/cut 0.0 3.695",
                "pair_coeff 1*4 5*8 lj/cut 0.0103 3.405",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation.
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[0][3],
                                      'B': self.original_masses[1][0],
                                      'C': self.original_masses[1][1],
                                      'N': self.original_masses[1][2],
                                      'O': self.original_masses[1][3]}
            
            # Set up the LAMMPS calculator with the specified commands.
            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                            #    log_file='log_IL.txt',
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
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.propagate(atoms, properties, system_changes, 0)
            self.tmd_calculate_intralayer(atoms)
            self.calculate_interlayer(atoms)
            atoms.set_chemical_symbols(self.original_atom_types)

        if self.system_type == 'BP':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.propagate(atoms, properties, system_changes, 0)
            self.calculate_intralayer(atoms)
            self.calculate_interlayer(atoms)
            atoms.set_chemical_symbols(self.original_atom_types)

    def calculate_intralayer(self, atoms):
        # First Layer
        atom_L1 = atoms[
            atoms.arrays["atom_types"] < len(self.chemical_symbols[0])
            ]
        atom_L1.numbers = atom_L1.arrays["atom_types"] + 1

        lammps_L1 = MonolayerLammpsCalculator(atom_L1,
                                              chemical_symbols=self.chemical_symbols[0],
                                              system_type=self.system_type)
        atom_L1.calc = lammps_L1
        L1_energy = atom_L1.get_potential_energy()
        L1_forces = atom_L1.get_forces()

        # Repeat similar calculations for the second layer.

        atom_L2 = atoms[
            atoms.arrays["atom_types"] >= len(self.chemical_symbols[0])
            ]
        atom_L2.numbers = atom_L2.arrays["atom_types"] - len(self.chemical_symbols[0]) + 1
        atom_L2.arrays['atom_types'] -= len(self.chemical_symbols[0])

        lammps_L2 = MonolayerLammpsCalculator(atom_L2,
                                              chemical_symbols=self.chemical_symbols[1],
                                              system_type=self.system_type)
        atom_L2.calc = lammps_L2
        L2_energy = atom_L2.get_potential_energy()
        L2_forces = atom_L2.get_forces()

        self.results["L1_energy"] = L1_energy
        self.results["L1_forces"] = L1_forces
        self.results["L2_energy"] = L2_energy
        self.results["L2_forces"] = L2_forces
        
    def tmd_calculate_intralayer(self, atoms):
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


        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': L1_atom_masses[0],
                            'He': L1_atom_masses[1],
                            'Li': L1_atom_masses[2]}
        # Set up and run the LAMMPS simulation for the first layer.
        lammps_L1 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses,
                            #   log_file='log_L1.txt',
                              keep_alive = False)
        atom_L1.calc = lammps_L1
        L1_energy = atom_L1.get_potential_energy()
        L1_forces = atom_L1.get_forces()

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

        atom_types = {'H': 1, 'He': 2, 'Li': 3}
        atom_type_masses = {'H': L2_atom_masses[0],
                            'He': L2_atom_masses[1],
                            'Li': L2_atom_masses[2]}
        lammps_L2 = LAMMPSlib(lmpcmds=cmds,
                              atom_types=atom_types,
                              atom_type_masses=atom_type_masses,
                            #   log_file='log_L2.txt'
                              )
        atom_L2.calc = lammps_L2
        
        L2_energy = atom_L2.get_potential_energy()
        L2_forces = atom_L2.get_forces()

        self.results["L1_energy"] = L1_energy
        self.results["L1_forces"] = L1_forces
        self.results["L2_energy"] = L2_energy
        self.results["L2_forces"] = L2_forces

    def calculate_interlayer(self, atoms):

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
                    f"minimize        1.0e-6 1.0e-8 10000 10000",
                    "write_data      lammps.dat_min"]

        self.set(amendments=min_cmds)
        atoms.numbers = atoms.arrays["atom_types"] + 1
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions"],
                       n_steps=1)
        self.calculate_intralayer(atoms)
        self.calculate_interlayer(atoms)
        atoms.set_chemical_symbols(self.original_atom_types)

    def clean_atoms(self, atoms):
        atoms.set_chemical_symbols(self.original_atom_types)
        if self.started:
            self.lmp.close()
            self.started = False
            self.initialized = False
            self.lmp = None

class BilayerLammpsCalculator(LAMMPSlib):
    implemented_properties = ["energy",
                              "energies",
                              "forces", "free_energy", "stress"]

    def __init__(self, atoms, chemical_symbols, system_type='TMD',**kwargs):

        Calculator.__init__(self, **kwargs)

        self.system_type = system_type
        self.original_atom_types = atoms.get_chemical_symbols()
        self.chemical_symbols = chemical_symbols

        self.original_masses = [[atomic_masses[atomic_numbers[n]] for n in t] for t in self.chemical_symbols]
        
        if self.system_type is None: 
            print("Specify type of bilayer. Options are 'TMD' or 'graphene'")

        if self.system_type == 'TMD':
            
            cmds = [
                # LAMMPS commands go here.
                "pair_style hybrid/overlay sw sw kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 kolmogorov/crespi/z 14.0 lj/cut 10.0",
                f"pair_coeff * * sw 1 tmd.sw {self.chemical_symbols[0][0]} {self.chemical_symbols[0][1]} {self.chemical_symbols[0][2]} NULL NULL NULL",
                f"pair_coeff * * sw 2 tmd.sw NULL NULL NULL {self.chemical_symbols[1][0]} {self.chemical_symbols[1][1]} {self.chemical_symbols[1][2]}",
                f"pair_coeff 1 6 kolmogorov/crespi/z 1 WS.KC  {self.chemical_symbols[0][0]} NULL NULL NULL NULL  {self.chemical_symbols[1][2]}",
                f"pair_coeff 2 4 kolmogorov/crespi/z 2 WS.KC NULL  {self.chemical_symbols[0][1]} NULL {self.chemical_symbols[1][0]} NULL NULL",
                f"pair_coeff 2 6 kolmogorov/crespi/z 3 WS.KC NULL  {self.chemical_symbols[0][1]} NULL NULL NULL  {self.chemical_symbols[1][2]}",
                f"pair_coeff 1 4 kolmogorov/crespi/z 4 WS.KC  {self.chemical_symbols[0][0]} NULL NULL {self.chemical_symbols[1][0]} NULL NULL",
                "pair_coeff * * lj/cut 0.0 3.0",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            # Define fixed atom types and masses for the simulation.

            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[1][0],
                                      'B': self.original_masses[1][1],
                                      'C': self.original_masses[1][2]} 
            # Set up the LAMMPS calculator with the specified commands.

            LAMMPSlib.__init__(self,
                      lmpcmds=cmds,
                      atom_types=fixed_atom_types,
                      atom_type_masses=fixed_atom_type_masses,
                      keep_alive = True)
 
        elif self.system_type == 'BP':
            # Define LAMMPS commands for the BP system.
            cmds = [
                "pair_style hybrid/overlay sw sw lj/cut 21.0",
                # f"pair_coeff * * sw bp.sw {chemical_symbols[0]} {chemical_symbols[1]} {chemical_symbols[2]} {chemical_symbols[3]}",
                f"pair_coeff * * sw 1 bp.sw T T B B NULL NULL NULL NULL",
                f"pair_coeff * * sw 2 bp.sw NULL NULL NULL NULL T T B B",
                "pair_coeff * * lj/cut 0.0 3.695",
                "pair_coeff 1*4 5*8 lj/cut 0.0103 3.405",
                "neighbor        2.0 bin",
                "neigh_modify every 1 delay 0 check yes"]
            
            # Define fixed atom types and masses for the simulation.
            fixed_atom_types = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}
            fixed_atom_type_masses = {'H': self.original_masses[0][0],
                                      'He': self.original_masses[0][1],
                                      'Li': self.original_masses[0][2],
                                      'Be': self.original_masses[0][3],
                                      'B': self.original_masses[1][0],
                                      'C': self.original_masses[1][1],
                                      'N': self.original_masses[1][2],
                                      'O': self.original_masses[1][3]}
            
            # Set up the LAMMPS calculator with the specified commands.
            LAMMPSlib.__init__(self,
                               lmpcmds=cmds,
                               atom_types=fixed_atom_types,
                               atom_type_masses=fixed_atom_type_masses,
                            #    log_file='log_IL.txt',
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

        Calculator.calculate(self, atoms, properties, system_changes)

        if self.system_type == 'TMD':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.propagate(atoms, properties, system_changes, 0)
            self.calculate_intralayer(atoms)
            self.calculate_interlayer(atoms)
            atoms.set_chemical_symbols(self.original_atom_types)

        if self.system_type == 'BP':
            atoms.numbers = atoms.arrays["atom_types"] + 1
            self.calculate_intralayer(atoms)
            self.calculate_interlayer(atoms)
            atoms.set_chemical_symbols(self.original_atom_types)

        L1_cond = atoms.arrays["atom_types"] < len(self.chemical_symbols[0])
        L2_cond = atoms.arrays["atom_types"] >= len(self.chemical_symbols[0])

        self.results["energy"] = self.results["L1_energy"] + self.results["L2_energy"] + self.results["IL_energy"]
        self.results["forces"] = self.results["IL_forces"]
        self.results["forces"][L1_cond] += self.results["L1_forces"]
        self.results["forces"][L2_cond] += self.results["L2_forces"]
        self.results["stress"] = self.results["L1_stress"] + self.results["L2_stress"] + self.results["IL_stress"]
        self.results['free_energy'] = self.results['energy']

    def calculate_intralayer(self, atoms):
        # First Layer
        atom_L1 = atoms[
            atoms.arrays["atom_types"] < len(self.chemical_symbols[0])
            ]
        atom_L1.numbers = atom_L1.arrays["atom_types"] + 1

        lammps_L1 = MonolayerLammpsCalculator(atom_L1,
                                              layer_symbols=self.chemical_symbols[0],
                                              system_type=self.system_type)
        atom_L1.calc = lammps_L1
        L1_energy = atom_L1.get_potential_energy()
        L1_stress = atom_L1.calc.results['stress']
        L1_forces = atom_L1.calc.results['forces']

        # Repeat similar calculations for the second layer.

        atom_L2 = atoms[
            atoms.arrays["atom_types"] >= len(self.chemical_symbols[0])
            ]
        atom_L2.numbers = atom_L2.arrays["atom_types"] - len(self.chemical_symbols[0]) + 1
        atom_L2.arrays['atom_types'] -= len(self.chemical_symbols[0])

        lammps_L2 = MonolayerLammpsCalculator(atom_L2,
                                              layer_symbols=self.chemical_symbols[1],
                                              system_type=self.system_type)
        atom_L2.calc = lammps_L2
        L2_energy = atom_L2.get_potential_energy()
        L2_stress = atom_L2.calc.results['stress']
        L2_forces = atom_L2.calc.results['forces']
        
        self.results["L1_energy"] = L1_energy
        self.results["L1_forces"] = L1_forces
        self.results["L1_stress"] = L1_stress
        self.results["L2_energy"] = L2_energy
        self.results["L2_forces"] = L2_forces
        self.results["L2_stress"] = L2_stress

    def calculate_interlayer(self, atoms):
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.numbers = tmp_atoms.arrays["atom_types"] + 1

        lammps_calc = InterlayerLammpsCalculator(atoms,
                                                 layer_symbols=self.chemical_symbols,
                                                 system_type=self.system_type)   
        tmp_atoms.calc = lammps_calc

        self.results['IL_energy'] = tmp_atoms.get_potential_energy()
        self.results['IL_forces'] = tmp_atoms.calc.results['forces']
        self.results['IL_stress'] = tmp_atoms.calc.results['stress']

    def minimize(self, atoms, method="fire", file = 'relax.lammps.traj.xyz'):
        min_cmds = [# Create a dump file for the trajectories
                    f"dump mydump all xyz 1 {file}",
                    f"min_style       {method}",
                    f"minimize       0 1.0e-6 1320 1320",
                    "write_data      lammps.dat_min",
                    "undump mydump"]

        self.set(amendments=min_cmds)
        atoms.numbers = atoms.arrays["atom_types"] + 1
        self.propagate(atoms,
                       properties=["energy",
                                   'forces',
                                   'stress'],
                       system_changes=["positions"],
                       n_steps=1)
        self.calculate_intralayer(atoms)
        self.calculate_interlayer(atoms)
        atoms.set_chemical_symbols(self.original_atom_types)
