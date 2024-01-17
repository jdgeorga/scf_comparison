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
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(
        self,
        max_num_layers: int,
        atom_types: List[str],
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
        atom_type_str: Pattern = compile("([A-Za-z]+)L([0-9]+)")
        atom_type_info: List[Dict[str, Dict[str, Union[int, str]]]] = []

        for i, atom_type in enumerate(atom_types):
            tmp: Match = atom_type_str.match(atom_type)
            if tmp is None:
                raise CalculatorSetupError("Invalid atom type format")
            relevant_info_tmp: Tuple[str] = tmp.groups()
            layer_id = int(relevant_info_tmp[1])
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
        self.layer_at_info: List[Tuple[int]] = [item for item in layer_at_info if len(item) != 0]
        self.num_layers = len(self.layer_at_info)
        self.device = device
        self.results = {}

    def setup_models(
        self,
        num_layers: int,
        intalayer_model_path_list: List[Path],
        interlayer_model_path_list: List[Path],
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

    def calculate(
        self,
        atoms: Atoms,
        properties=None,
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms)
        intralayer_chemical_symbol_to_type = {"Mo": 0, "S": 1}
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
                        atoms=tmp_at,
                        r_max=r_max,
                    )
                    for k in AtomicDataDict.ALL_ENERGY_KEYS:
                        if k in data:
                            del data[k]
                    at_num = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    valid_atomic_numbers = [
                        atomic_numbers[sym] for sym in chemical_symbol_to_type
                    ]
                    _min_Z = min(valid_atomic_numbers)
                    _max_Z = max(valid_atomic_numbers)
                    Z_to_index = full(
                        size=(1 + _max_Z - _min_Z,),
                        fill_value=-1,
                        dtype=long,
                    )
                    for sym, typeid in chemical_symbol_to_type.items():
                        Z_to_index[atomic_numbers[sym] - _min_Z] = typeid
                    del data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
                    data[AtomicDataDict.ATOM_TYPE_KEY] = Z_to_index.to(
                        device=self.device
                    )[at_num - _min_Z]
                    data = data.to(self.device)
                    data = AtomicData.to_AtomicDataDict(data)
                    out = model(data)
                    results = get_results_from_model_out(out)
                    energies[rel_ats] += results["energies"]
                    forces[rel_ats] += results["forces"]
                    energy += results["energy"]
                    
                    if l_id == 0:
                        L1_energy += results["energy"]
                        L1_forces[rel_ats] += results["forces"]
                    if l_id == 1:
                        L2_energy += results["energy"]
                        L2_forces[rel_ats] += results["forces"]

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
                    out = model(data)
                    results = get_results_from_model_out(out)
                    energies[rel_ats] += results["energies"]
                    forces[rel_ats] += results["forces"]
                    energy += results["energy"]
                    
                    IL_forces[rel_ats] += results["forces"]
                    IL_energy += results["energy"]

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


if __name__ == "__main__":
    from ase.io import read

    calc = NeuqIPMultiLayerCalculator(4, ["MoL1", "SL1", "SeL1", "MoL2", "SL2", "SeL2"])
    calc.setup_models(
        2,
        [Path("./intralayer.pth"), Path("./intralayer.pth")],
        [Path("./interlayer.pth")],
    )
    at = read("./MoS2-Bilayer.xyz")
    calc.calculate(at)

# Example Usage:
# calc = AllegroCalculator(4, 
#                          ["MoL1", "SL1", "SeL1", "MoL2", "SL2", "SeL2"],
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
