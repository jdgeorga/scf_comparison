from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase import Atoms
import numpy as np


class NLayerCalculator(Calculator):
    implemented_properties = ['energy', 'energies', 'forces']  # Define the properties this calculator can compute
    
    def __init__(self,
                 atoms,
                 intralayer_calculators: list[Calculator],
                 interlayer_calculators: list[Calculator],
                 layer_symbols: list[str],
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.intra_calc_list = intralayer_calculators
        self.inter_calc_list = interlayer_calculators
        self.layer_symbols = layer_symbols
        

    def calculate(self,
                  atoms: Atoms,
                  properties=None,
                  system_changes=all_changes):
        
        if properties is None:
            properties = self.implemented_properties

        self.atoms_layer_list = []

        Calculator.calculate(self, atoms, properties, system_changes)

        self.get_atoms_layer_list(self.atoms)

        self.results['layer_energy'] = np.zeros((len(self.intra_calc_list), len(self.intra_calc_list)))
        self.results['layer_forces'] = np.zeros((len(self.intra_calc_list),
                                                 len(self.intra_calc_list),
                                                 atoms.get_global_number_of_atoms(), 3))
        # self.results['layer_energies'] = np.zeros((len(self.intra_calc_list),
        #                                            len(self.intra_calc_list),
        #                                            atoms.get_global_number_of_atoms()))
        for layer_index_i in range(len(self.intra_calc_list)):
            for layer_index_j in range(len(self.intra_calc_list)):    
                if layer_index_i == layer_index_j:
                    self.calculate_intralayer(atoms, layer=layer_index_i)
                elif layer_index_i < layer_index_j:
                    self.calculate_interlayer(atoms,
                                              layer_1=layer_index_i,
                                              layer_2=layer_index_j)
                elif layer_index_i > layer_index_j:
                    self.results['layer_energy'][layer_index_i, layer_index_j] = self.results['layer_energy'][layer_index_j, layer_index_i]
                    self.results['layer_forces'][layer_index_i, layer_index_j] = self.results['layer_forces'][layer_index_j, layer_index_i]
                    # self.results['layer_energies'][layer_index_i, layer_index_j] = self.results['layer_energies'][layer_index_j, layer_index_i]
        self.results["energy"] = 1./2 * (self.results["layer_energy"].sum() + self.results["layer_energy"].trace())
        self.results["forces"] = 1./2 * (self.results["layer_forces"].sum(axis=(0,1)) + self.results["layer_forces"].trace(axis1 = 0, axis2 = 1))
        # self.results['energies'] = 1./2 * (self.results["layer_energies"].sum(axis=(0,1)) + self.results["layer_energies"].trace(axis1 = 0, axis2 = 1))
     
    def calculate_intralayer(self, atoms, layer: int):

        if layer <= len(self.intra_calc_list):
            calc = self.intra_calc_list[layer]
            atoms_L = self.atoms_layer_list[layer]
        else:
            raise ValueError("layer number muls")
        
        atoms_L.calc = calc
        atoms_L.calc.calculate(atoms_L)

        lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:(layer)]])
        layer_atom_indices = [lower_layers_num_atoms,
                              lower_layers_num_atoms + len(self.atoms_layer_list[layer])]
        
        self.results[f"layer_energy"][layer,layer] = atoms_L.calc.results['energy']
        self.results[f"layer_forces"][layer,layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']
        # self.results['layer_energies'][layer,layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['energies']

    def calculate_interlayer(self, atoms, layer_1: int, layer_2: int):
        # JDG: code only considers adjacent layers
        if layer_2 > layer_1 + 1:
            return 0
        
        atoms_L = self.atoms_layer_list[layer_1].copy() + self.atoms_layer_list[layer_2].copy()
        calc = self.inter_calc_list[layer_1]
        atoms_L.calc = calc
        atoms_L.calc.calculate(atoms_L)

        lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:(layer_1)]])
        layer_atom_indices = [lower_layers_num_atoms,
                              lower_layers_num_atoms + len(self.atoms_layer_list[layer_1]) + len(self.atoms_layer_list[layer_2])]
        
        self.results["layer_energy"][layer_1,layer_2] = atoms_L.calc.results['energy']
        self.results["layer_forces"][layer_1,layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']
        # self.results['layer_energies'][layer_1,layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.get_potential_energies()

    def get_atoms_layer_list(self, atoms):

        for layer in range(len(self.intra_calc_list)):
        
            lower_layer_num_atoms = sum([len(layer_atoms) for layer_atoms in self.layer_symbols[:(layer)]])
            layer_atom_indices = [lower_layer_num_atoms,
                                  lower_layer_num_atoms + len(self.layer_symbols[layer]) - 1]
            
            atoms_L = atoms.copy()[np.logical_and(atoms.arrays["atom_types"] <= layer_atom_indices[1],
                                                  atoms.arrays["atom_types"] >= layer_atom_indices[0])]
            
            self.atoms_layer_list.append(atoms_L)
