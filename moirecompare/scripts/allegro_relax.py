from ase.io import read, write
import numpy as np
from moirecompare.calculators import AllegroCalculator, NLayerCalculator
from ase.optimize import FIRE, BFGS
from ase.io.trajectory import Trajectory
import time

def allegro_relax(input_file,
                  output_file,
                  intralayer_model,
                  interlayer_model,
                  layer_symbols,
                  force_tol=4e-3):
    """
    Perform relaxation of a bilayer TMD system using AllegroCalculator.

    Parameters:
    input_file (str): Path to the input XYZ file containing the atomic structure.
    output_file (str): Path prefix for the output trajectory and XYZ files.
    intralayer_model (str): Path to the intralayer model file.
    interlayer_model (str): Path to the interlayer model file.
    force_tol (float): Force tolerance for the relaxation convergence in eV/Å.
    """
    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")

    # Separate the atoms into two layers based on atom types
    at_1 = atoms[atoms.arrays['atom_types'] < 3]
    at_2 = atoms[atoms.arrays['atom_types'] >= 3]

    # Initialize calculators for intralayer interactions
    intra_calc_1 = AllegroCalculator(at_1, layer_symbols[0], model_file=intralayer_model, device='cuda')
    intra_calc_2 = AllegroCalculator(at_2, layer_symbols[1], model_file=intralayer_model, device='cuda')

    # Initialize calculator for interlayer interactions
    inter_calc = AllegroCalculator(atoms, layer_symbols, model_file=interlayer_model, device='cuda')

    # Combine the calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms, [intra_calc_1, intra_calc_2], [inter_calc], layer_symbols, device='cuda')

    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc
    atoms.calc.calculate(atoms)

    # Print unrelaxed energy information
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n"
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Perform the relaxation using the FIRE algorithm
    dyn = FIRE(atoms, trajectory=f"{output_file}.traj")
    dyn.run(fmax=force_tol)

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

    # Input and output file paths and model files
    input_file = "MoS2_1D_3.4deg_30atoms.xyz"
    output_file = "MoS2_1D_3.4deg_30atoms_allegro"
    intralayer_MoS2_model = "mos2_intra_no_shift.pth"
    interlayer_MoS2_model = "interlayer_beefy.pth"

    # Define the symbols for the layers
    layer_symbols = [["Mo", "S", "S"], ["Mo", "S", "S"]]

    # Run the relaxation process
    allegro_relax(input_file,
                  output_file,
                  intralayer_MoS2_model,
                  interlayer_MoS2_model,
                  layer_symbols)

    # Optional: Uncomment and modify the following lines to run multiple relaxations
    # for j in ["compress", "stretch"]:
    #     for i in [10, 25, 50, 100]:
    #         allegro_relax(f"mos2_1D_{i}_{j}.xyz", f"allegro_relaxed/mos2_1D_{i}_{j}_allegro", intralayer_MoS2_model, interlayer_MoS2_model)

    end_time = time.time() - start_time
    print(f"Total time: {end_time:.2f} seconds")
