from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from warnings import warn
from typing import List, Dict, Union, Tuple, Optional
from re import compile, match, Pattern, Match
from torch.jit import ScriptModule
from torch import full, long
import torch
from pathlib import Path
from ase.atoms import Atoms
from itertools import combinations
from numpy import zeros, where, array, logical_and
import numpy as np


def get_results_from_model_out(model_out):
    results = {}
    if AtomicDataDict.TOTAL_ENERGY_KEY in model_out:
        results["energy"] = (
            model_out[AtomicDataDict.TOTAL_ENERGY_KEY]
            .detach()
            .cpu()
            .numpy()
            .reshape(tuple())
        )
        results["free_energy"] = results["energy"]
    if AtomicDataDict.PER_ATOM_ENERGY_KEY in model_out:
        results["energies"] = (
            model_out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            .detach()
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    if AtomicDataDict.FORCE_KEY in model_out:
        results["forces"] = model_out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
    return results

class AllegroCalculator(Calculator):
    # Define the properties that the calculator can handle
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self,
                 atoms,
                 layer_symbols: list[str],
                 model_file: str,
                 device='cpu',
                 **kwargs):
        """
        Initializes the AllegroCalculator with a given set of atoms, layer symbols, model file, and device.

        :param atoms: ASE atoms object.
        :param layer_symbols: List of symbols representing different layers in the structure.
        :param model_file: Path to the file containing the trained model.
        :param device: Device to run the calculations on, default is 'cpu'.
        :param kwargs: Additional keyword arguments for the base class.
        """
        self.atoms = atoms  # ASE atoms object
        self.atom_types = atoms.arrays['atom_types']  # Extract atom types from atoms object
        self.device = device  # Device for computations

        # Flatten the layer symbols list
        self.layer_symbols = [symbol for sublist in layer_symbols for symbol in (sublist if isinstance(sublist, list) else [sublist])]

        # Load the trained model and metadata
        self.model, self.metadata_dict = load_deployed_model(model_path=model_file, device=device)
        # print(self.metadata_dict['n_species'],len(self.layer_symbols))
        # if int(self.metadata_dict['n_species']) != len(self.layer_symbols):
        #     raise ValueError("Mismatch between the number of atom types in model and provided layer symbols.",
        #                      "Are you using an intralayer or interlayer model?")
        
        # Determine unique atom types and their indices
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        # Ensure the number of unique atom types matches the number of layer symbols provided
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("Mismatch between the number of atom types and provided layer symbols.")

        # Initialize the base Calculator class with any additional keyword arguments
        Calculator.__init__(self, **kwargs)

    def calculate(self,
                  atoms,
                  properties=None,
                  system_changes=all_changes):
        """
        Performs the calculation for the given atoms and properties.

        :param atoms: ASE atoms object to calculate properties for.
        :param properties: List of properties to calculate. If None, uses implemented_properties.
        :param system_changes: List of changes that have been made to the system since last calculation.
        """
        # Default to implemented properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Create a temporary copy of the atoms object
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.calc = None  # Remove any attached calculator

        r_max = self.metadata_dict["r_max"]  # Maximum radius for calculations

        # Backup original atomic numbers and set new atomic numbers based on relative layer types
        original_atom_numbers = tmp_atoms.numbers.copy()
        tmp_atoms.set_atomic_numbers(self.relative_layer_types + 1)
        tmp_atoms.arrays['atom_types'] = self.relative_layer_types

        # Prepare atomic data for the model
        data = AtomicData.from_ase(atoms=tmp_atoms, r_max=r_max, include_keys=[AtomicDataDict.ATOM_TYPE_KEY])

        # Remove energy keys from the data if present
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]

        # Move data to the specified device and convert to AtomicDataDict format
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # Pass data through the model to get the output
        out = self.model(data)

        # Restore the original atomic numbers and types
        tmp_atoms.set_atomic_numbers(original_atom_numbers)
        tmp_atoms.arrays['atom_types'] = self.atom_types
        
        # Process the model output to get the desired results
        self.results = get_results_from_model_out(out)


# 
class AllegroCalculator_old(Calculator):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(
        self,
        max_num_layers: int,
        atom_types: List[str],
        intralayer_symbol_type: Dict[str, int],
        device: Union[str, torch.device],
        model_dictionary: Dict[str, ScriptModule] = None,
    ):
        Calculator.__init__(self)
        num_atom_types_tmp = len(atom_types)
        tmp_atom_types = list(set(atom_types))
        num_atom_types = len(tmp_atom_types)
        layer_at_info: List[Tuple[int]] = [()] * max_num_layers
        if num_atom_types != num_atom_types_tmp:
            raise CalculatorSetupError("You have repeated atom types")
        atom_type_str: Pattern = compile("([A-Za-z]+)([0-9]+)L([0-9]+)")
        atom_type_info: List[Dict[str, Dict[str, Union[int, str]]]] = []
 
        for i, atom_type in enumerate(atom_types):
            tmp: Match = atom_type_str.match(atom_type)
            if tmp is None:
                raise CalculatorSetupError("Invalid atom type format")
            relevant_info_tmp: Tuple[str] = tmp.groups()
            layer_id = int(relevant_info_tmp[2])
            layer_at_info[layer_id - 1] += (i,)
            if layer_id > max_num_layers:
                raise CalculatorSetupError("Layer ID exceeds max number of layers")
            try:
                ase_symbol_id = chemical_symbols.index(relevant_info_tmp[0])
            except ValueError:
                raise CalculatorSetupError("Invalid chemical symbol")
            ase_symbol = chemical_symbols[ase_symbol_id]
            ase_at_mass = atomic_masses[ase_symbol_id]
            ase_at_num = atomic_numbers[ase_symbol]

            atom_info_dict: Dict[str, Dict[str, Union[int, str]]] = {
                atom_type: {
                    "Z": ase_at_num,
                    "mass": ase_at_mass,
                    "symbol": ase_symbol,
                    "layer_id": layer_id,
                }
            }
            atom_type_info.append(atom_info_dict)
        self.atom_type_info: List[Dict[str, Dict[str, Union[int, str]]]] = atom_type_info
        self.atom_types: List[str] = atom_types
        self.intralayer_symbol_type = intralayer_symbol_type
        self.layer_at_info: List[Tuple[int]] = [item for item in layer_at_info if len(item) != 0]
        self.num_layers = len(self.layer_at_info)
        self.device = device
        self.results = {}

    def setup_models(
        self,
        num_layers: int,
        intalayer_model_path_list: List[Path],
        interlayer_model_path_list: List[Path],
        IL_factor: float = 1.0,
        L1_factor: float = 1.0,
        L2_factor: float = 1.0,
    ):
        intralayer_model_dict: Dict[int, (ScriptModule, float)] = {}
        interlayer_model_dict: Dict[str, (ScriptModule, float)] = {}
        if len(intalayer_model_path_list) != num_layers:
            raise CalculatorSetupError("Invalid number of intralayer models")
        if len(interlayer_model_path_list) != num_layers * (num_layers - 1) // 2:
            raise CalculatorSetupError("Invalid number of interlayer models")
        ct = 0
        for l1 in range(0, num_layers):
            try:
                intra_model, tmp = load_deployed_model(
                    model_path=intalayer_model_path_list[l1],
                    device=self.device
                )
            except AttributeError:
                intra_model = None
                tmp = {R_MAX_KEY: 0}
            intralayer_model_dict.update({l1: (intra_model, float(tmp[R_MAX_KEY]))})
            if l1 != num_layers - 1:
                for l2 in range(1, num_layers):
                    try:
                        inter_model, tmp = load_deployed_model(
                            model_path=interlayer_model_path_list[ct],
                            device=self.device
                        )
                    except AttributeError:
                        inter_model = None
                        tmp = {R_MAX_KEY: 0}
                    key = f"{l1}_{l2}"
                    interlayer_model_dict.update(
                        {key: (inter_model, float(tmp[R_MAX_KEY]))}
                    )
                    ct += 1
        self.intralayer_model_dict: Dict[int, (ScriptModule, float)] = intralayer_model_dict
        self.interlayer_model_dict: Dict[str, (ScriptModule, float)] = interlayer_model_dict
        self.IL_factor = IL_factor
        self.L1_factor = L1_factor
        self.L2_factor = L2_factor

    def calculate(
        self,
        atoms: Atoms,
        properties=None,
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms)
        intralayer_chemical_symbol_to_type = self.intralayer_symbol_type
        if properties is None:
            properties = self.implemented_properties
        if AtomicDataDict.ATOM_TYPE_KEY not in atoms.arrays:
            raise CalculatorError("Atoms object must have array ATOM_TYPE_KEY")

        n_atoms = atoms.get_global_number_of_atoms()
        forces = zeros((n_atoms, 3))
        energies = zeros(n_atoms)
        energy = 0

        L1_energy = 0
        L1_forces = zeros((n_atoms, 3))
        L2_energy = 0
        L2_forces = zeros((n_atoms, 3))  
        IL_energy = 0
        IL_forces = zeros((n_atoms, 3))

        for l_id, l in enumerate(self.layer_at_info):
            model, r_max = self.intralayer_model_dict[l_id]
            if model is not None:
                all_types = atoms.arrays[AtomicDataDict.ATOM_TYPE_KEY]
                types_at = list(l)
                min_id = min(types_at)
                max_id = max(types_at)
                rel_ats = where(logical_and(all_types >= min_id, all_types <= max_id))[0]
                if len(rel_ats) != 0:
                    chemical_symbol_to_type = intralayer_chemical_symbol_to_type
                    
                    tmp_at = atoms[rel_ats].copy()
                    data = AtomicData.from_ase(
                        atoms=tmp_at, r_max=r_max, include_keys=[AtomicDataDict.ATOM_TYPE_KEY]
                        )
                    # for k in AtomicDataDict.ALL_ENERGY_KEYS:
                    #     if k in data:
                    #         del data[k]
                    # at_num = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    # valid_atomic_numbers = [
                    #     atomic_numbers[sym] for sym in chemical_symbol_to_type
                    # ]
                    # _min_Z = min(valid_atomic_numbers)
                    # _max_Z = max(valid_atomic_numbers)
                    # Z_to_index = full(
                    #     size=(1 + _max_Z - _min_Z,),
                    #     fill_value=-1,
                    #     dtype=long,
                    # )
                    # for sym, typeid in chemical_symbol_to_type.items():
                    #     Z_to_index[atomic_numbers[sym] - _min_Z] = typeid
                    # del data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    # data[AtomicDataDict.ATOM_TYPE_KEY] = Z_to_index.to(
                    #     device=self.device
                    # )[at_num - _min_Z]

                    for k in AtomicDataDict.ALL_ENERGY_KEYS:
                        if k in data:
                            del data[k]

                    
                    data = data.to(self.device)
                    data = AtomicData.to_AtomicDataDict(data)
                    # print("ALLEGRO_OG",data)
                    out = model(data)
                    results = get_results_from_model_out(out)
                    print("Old", results)


                    
                    if l_id == 0:
                        L1_energy += results["energy"] * self.L1_factor
                        L1_forces[rel_ats] += results["forces"] * self.L1_factor
                        energies[rel_ats] += results["energies"] * self.L1_factor
                        forces[rel_ats] += results["forces"] * self.L1_factor
                        energy += results["energy"] * self.L1_factor
                    if l_id == 1:
                        L2_energy += results["energy"] * self.L2_factor
                        L2_forces[rel_ats] += results["forces"] * self.L2_factor
                        energies[rel_ats] += results["energies"] * self.L2_factor
                        forces[rel_ats] += results["forces"] * self.L2_factor
                        energy += results["energy"] * self.L2_factor

        for l_id1, l_id2 in combinations(range(len(self.layer_at_info)), 2):
            l1 = self.layer_at_info[l_id1]
            l2 = self.layer_at_info[l_id2]
            key = f"{l_id1}_{l_id2}"
            model, r_max = self.interlayer_model_dict[key]
            if model is not None:
                all_types = atoms.arrays[AtomicDataDict.ATOM_TYPE_KEY]
                types_at_l1 = list(l1)
                min_id_l1 = min(types_at_l1)
                max_id_l1 = max(types_at_l1)
                rel_ats_l1 = where(
                    logical_and(all_types >= min_id_l1, all_types <= max_id_l1)
                )[0]
                types_at_l2 = list(l2)
                min_id_l2 = min(types_at_l2)
                max_id_l2 = max(types_at_l2)
                rel_ats_l2 = where(
                    logical_and(all_types >= min_id_l2, all_types <= max_id_l2)
                )[0]
                rel_ats = array(list(rel_ats_l1) + list(rel_ats_l2))
                if len(rel_ats) != 0:
                    tmp_at = atoms[rel_ats].copy()
                    types_at_for_model = [tmp - min_id_l1 for tmp in types_at_l1]
                    types_at_for_model += [
                        tmp - min_id_l2 + max_id_l1 + 1 for tmp in types_at_l2
                    ]
                    tmp_at.set_atomic_numbers(tmp_at.arrays["atom_types"] + 1)
                    data = AtomicData.from_ase(
                        atoms=tmp_at,
                        r_max=r_max,
                        include_keys=[AtomicDataDict.ATOM_TYPE_KEY],
                    )
                    for k in AtomicDataDict.ALL_ENERGY_KEYS:
                        if k in data:
                            del data[k]
                    data = data.to(self.device)
                    data = AtomicData.to_AtomicDataDict(data)
                    # print("OG_INTERLAYERDATA",data)
                    out = model(data)
                    
                    results = get_results_from_model_out(out)
                    energies[rel_ats] += results["energies"] * self.IL_factor
                    forces[rel_ats] += results["forces"] * self.IL_factor
                    energy += results["energy"] * self.IL_factor
                    
                    IL_forces[rel_ats] += results["forces"] * self.IL_factor
                    IL_energy += results["energy"] * self.IL_factor

        self.results = {}
        self.results["energy"] = energy
        self.results["energies"] = energies
        self.results["forces"] = forces
        self.results["L1_energy"] = L1_energy
        self.results["L1_forces"] = L1_forces
        self.results["L2_energy"] = L2_energy
        self.results["L2_forces"] = L2_forces
        self.results["IL_energy"] = IL_energy
        self.results["IL_forces"] = IL_forces