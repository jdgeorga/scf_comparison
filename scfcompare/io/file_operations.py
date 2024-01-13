import os
from ase.io import read
from scfcompare.utils import calculate_xy_area, extract_total_wall_time_in_seconds, add_allegro_number_array

def extract_qe_data(directories, vals):
    """
    Extract forces, total energy, atomic positions, and atom types from QE output files in given directories.
    """
    data = {}
    print(f"Extracting QE data from directory")

    for directory, vs in zip(directories, vals):
        output_file = os.path.join(directory, 'relax.out')
        if os.path.exists(output_file):
            atoms = read(output_file, format='espresso-out')
            forces = atoms.get_forces()
            total_energy = atoms.get_total_energy()
            positions = atoms.get_positions()
            volume = atoms.get_volume()
            wall_time = extract_total_wall_time_in_seconds(output_file)
            area = calculate_xy_area(atoms)
            cell = atoms.cell

            # Add atom types using the add_allegro_number_array function
            atom_types = add_allegro_number_array(atoms,eps = 0.5, min_samples=20)
            
            # print(directory, wall_time)
            data[directory] = {
                'directory': directory,
                'conv_values': vs,
                'forces': forces,
                'total_energy': total_energy,
                'positions': positions,
                'atom_types': atom_types,  # Add atom types
                'wall_time': wall_time,
                'volume': volume,
                'area': area,
                'cell': cell
            }
        else:
            print(f"No output file found in {directory}")

    return data

def get_subdirectories(directory):
    """Get all subdirectories from the specified directory."""
    print(f"Getting subdirectories of directory: {directory}")

    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return sorted(subdirectories)