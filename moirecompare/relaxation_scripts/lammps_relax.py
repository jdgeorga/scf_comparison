from ase.io import read, write
import numpy as np
from moirecompare.calculators import NLayerCalculator
from moirecompare.calculators import MonolayerLammpsCalculator, InterlayerLammpsCalculator
from ase.optimize import FIRE, BFGS
from ase.io.trajectory import Trajectory

import time

def lammps_relax(input_file, output_file, layer_symbols):
    """
    Perform relaxation of a bilayer TMD system using LAMMPS calculators.

    Parameters:
    input_file (str): Path to the input XYZ file containing the atomic structure.
    output_file (str): Path prefix for the output trajectory and XYZ files.
    """
    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")

    # Adjust the z-positions and cell size to avoid interactions with periodic images
    atoms.positions[:, 2] += 15
    atoms.cell[2, 2] = 100

    # Separate the atoms into two layers based on atom types
    at_1 = atoms.copy()[atoms.arrays["atom_types"] < 3]
    at_2 = atoms.copy()[atoms.arrays["atom_types"] >= 3]

    # Initialize intralayer LAMMPS calculators
    intralayer_calcs = [
        MonolayerLammpsCalculator(atoms[atoms.arrays['atom_types'] < 3], layer_symbols[0], system_type='TMD', intra_potential='tmd.sw')
    ]
    interlayer_calcs = []

    # Initialize calculators for each layer and interlayer interactions
    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[
            np.logical_and(atoms.arrays['atom_types'] >= i*3, atoms.arrays['atom_types'] < (i+1)*3)
        ]
        intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms, layer_symbols=layer_symbols[i], system_type='TMD', intra_potential='tmd.sw'))
        
        bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i-1)*3, atoms.arrays['atom_types'] < (i+1)*3)]
        interlayer_calcs.append(
            InterlayerLammpsCalculator(bilayer_atoms, layer_symbols=layer_symbols[i-1:i+1], system_type='TMD')
        )

    # Combine the calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms, intralayer_calcs, interlayer_calcs, layer_symbols)

    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc
    atoms.calc.calculate(atoms)

    # Print unrelaxed energy information
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n"
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Perform the relaxation using the FIRE algorithm
    dyn = FIRE(atoms, trajectory=f'{output_file}.traj')
    dyn.run(fmax=1e-3)

    # Print relaxed energy information
    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n"
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Read the trajectory and write the final relaxed structure to an XYZ file
    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    images = [atom for atom in traj]
    write(f"{output_file}.traj.xyz", images, format="extxyz")

if __name__ == "__main__":
    start_time = time.time()

    # Input and output file paths
    input_file = "../MoS2_1D_3.4deg_30atoms.xyz"
    output_file = "MoS2_1D_3.4deg_30atoms_lammps"

    # Define the symbols for the layers
    layer_symbols = [["Mo", "S", "S"], ["Mo", "S", "S"]]


    # Run the relaxation process
    lammps_relax(input_file, output_file, layer_symbols)

    # Optional: Uncomment and modify the following lines to run multiple relaxations
    # for j in ["stretch"]:
    #     for i in [10, 25, 50, 100]:
    #         lammps_relax(f"mos2_1D_{i}_{j}.xyz", f"lammps_relaxed/mos2_1D_{i}_{j}_lammps")

    end_time = time.time() - start_time
    print(f"Total time: {end_time:.2f} seconds")
