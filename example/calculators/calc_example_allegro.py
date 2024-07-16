from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures
import numpy as np  # Import numpy for numerical operations
from moirecompare.calculators import AllegroCalculator, NLayerCalculator  # Import custom calculators from moirecompare package
from ase.optimize import FIRE, BFGS  # Import optimization algorithms from ASE

def allegro_relax(input_file, output_file):
    """
    Function to relax a given atomic structure using Allegro and NLayer calculators.
    
    Parameters:
    input_file (str): Path to the input structure file in extxyz format.
    output_file (str): Prefix for the output files to store relaxed structure and trajectory.
    """
    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")

    # Split the atoms into two groups based on atom types
    at_1 = atoms[atoms.arrays['atom_types'] < 3]
    at_2 = atoms[atoms.arrays['atom_types'] >= 3]

    # Define paths to the machine learning models
    intralayer_MoS2_model = "intralayer_beefy.pth"
    interlayer_MoS2_model = "interlayer_beefy_truncated.pth"

    # Define atomic symbols for each layer
    layer_symbols = [["Mo", "S", "S"],
                     ["Mo", "S", "S"]]

    # Set up intralayer calculators for each group of atoms
    intra_calc_1 = AllegroCalculator(at_1,
                                     layer_symbols[0],
                                     model_file=intralayer_MoS2_model,
                                     device='cpu') # change to 'cuda' if GPU is available
    intra_calc_2 = AllegroCalculator(at_2,
                                     layer_symbols[1],
                                     model_file=intralayer_MoS2_model,
                                     device='cpu') # change to 'cuda' if GPU is available

    # Set up interlayer calculator for the combined structure
    inter_calc = AllegroCalculator(atoms,
                                   layer_symbols,
                                   model_file=interlayer_MoS2_model,
                                   device='cpu') # change to 'cuda' if GPU is available

    # Combine the intra- and interlayer calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms,
                                    [intra_calc_1, intra_calc_2],
                                    [inter_calc],
                                    layer_symbols,
                                    device='cpu') # change to 'cuda' if GPU is available

    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc

    # Perform an initial calculation to get unrelaxed energy
    atoms.calc.calculate(atoms)
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Set up the FIRE optimizer for structural relaxation
    dyn = FIRE(atoms, trajectory=f"{output_file}.traj")
    dyn.run(fmax=4e-3)  # Run until the maximum force is below 4e-3 eV/Å

    # Print the relaxed energy
    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Import additional ASE modules for handling trajectories
    from ase.io.trajectory import Trajectory

    # Read the trajectory file generated during relaxation
    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    images = [atom for atom in traj]  # Collect all images from the trajectory

    # Write the final relaxed structure to an output file in extxyz format
    write(f"{output_file}.traj.xyz", images, format="extxyz")

if __name__ == "__main__":
    # Define the input XYZ file and output file prefix
    xyz_file_path = "MoS2-Bilayer.xyz"
    out_file = 'MoS2-Bilayer_allegro'

    # Call the relaxation function with specified input and output paths
    allegro_relax(xyz_file_path, out_file)