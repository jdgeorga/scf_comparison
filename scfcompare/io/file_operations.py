import os
from ase.io import read
from ase import Atoms
from scfcompare.utils import calculate_xy_area, extract_total_wall_time_in_seconds, add_allegro_number_array

# ADDING ATOMS implementation
def get_atoms_list_from_QE(directories, vals = []):
    atoms = {}
    for directory, conv_values in zip(directories, vals):
        output_file = os.path.join(directory, 'relax.out')
        if os.path.exists(output_file):
            atom: Atoms= read(output_file, format='espresso-out')

            atom_types = add_allegro_number_array(atoms,eps = 0.5, min_samples=20)
            wall_time = extract_total_wall_time_in_seconds(output_file)
            xy_area = calculate_xy_area(atom)

            atom.arrays['conv_values'] = conv_values
            atom.arrays['atom_types'] = atom_types
            atom.arrays['xy_area'] = xy_area
            atom.arrays['wall_time'] = wall_time

            atoms[directory] = atom
        else:
            print(f"No output file found in {directory}")

    return atoms

def get_subdirectories(directory):
    """Get all subdirectories from the specified directory."""
    print(f"Getting subdirectories of directory: {directory}")

    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return sorted(subdirectories)