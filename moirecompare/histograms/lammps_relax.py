from ase.io import read, write
import numpy as np
from moirecompare.calculators import NLayerCalculator
from moirecompare.calculators import (MonolayerLammpsCalculator, 
                                      InterlayerLammpsCalculator)
from ase.optimize import FIRE

def lammps_relax(input_file, output_file, layer_symbols):
    """
    Relax a structure using LAMMPS.

    Args:
    input_file (str): Path to the input structure file.
    output_file (str): Path to save the relaxed structure.
    layer_symbols (list of lists): Chemical symbols for each layer.
        Example: [["Mo", "S", "S"], ["W", "Se", "Se"]]

    Returns:
    ase.Atoms: Relaxed atomic structure.
    """
    atoms = read(input_file, format="extxyz")
    atoms.positions[:, 2] += 15
    atoms.cell[2, 2] = 100

    intralayer_calcs = [
        MonolayerLammpsCalculator(atoms[atoms.arrays['atom_types'] < 3],
                                  layer_symbols[0],
                                  system_type='TMD',
                                  intra_potential='tmd.sw')
    ]
    interlayer_calcs = []

    for i in range(1, len(layer_symbols)):
        layer_atoms = atoms[
            np.logical_and(atoms.arrays['atom_types'] >= i*3,
                           atoms.arrays['atom_types'] < (i+1)*3)
        ]
        intralayer_calcs.append(
            MonolayerLammpsCalculator(layer_atoms,
                                      layer_symbols=layer_symbols[i],
                                      system_type='TMD',
                                      intra_potential='tmd.sw')
        )
        
        bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i-1)*3,
                                             atoms.arrays['atom_types'] < (i+1)*3)]
        interlayer_calcs.append(
            InterlayerLammpsCalculator(bilayer_atoms,
                                       layer_symbols=layer_symbols[i-1:i+1],
                                       system_type='TMD')
        )

    n_layer_calc = NLayerCalculator(atoms,
                                    intralayer_calcs,
                                    interlayer_calcs,
                                    layer_symbols)

    atoms.calc = n_layer_calc 
    atoms.calc.calculate(atoms)

    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n"
          f"layer_energy {atoms.calc.results['layer_energy']}")

    dyn = FIRE(atoms, trajectory=f'{output_file}.traj')
    dyn.run(fmax=1e-3)

    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n"
          f"layer_energy {atoms.calc.results['layer_energy']}")
    
    from ase.io.trajectory import Trajectory

    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    images = []
    for atom in traj:
        images.append(atom)

    write(f"{output_file}.traj.xyz", images, format="extxyz")
    
    return atoms

def main():
    """
    Main function to demonstrate usage of lammps_relax.
    """
    input_file = "MoS2WSe2_2D_Stretch_14atoms.xyz"
    output_file = "MoS2WSe2_2D_Stretch_14atoms_lammps"
    layer_symbols = [["Mo", "S", "S"], ["W", "Se", "Se"]]
    relaxed_atoms = lammps_relax(input_file, output_file, layer_symbols)
    print(f"Relaxation completed. Output saved to {output_file}.traj.xyz")

if __name__ == "__main__":
    main()
