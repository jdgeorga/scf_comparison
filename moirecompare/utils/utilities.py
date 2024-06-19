"""
Utilities for your_package_name.

This module contains utility functions used across the package.
"""

import numpy as np
from ase import Atoms


def calculate_xy_area(atoms: Atoms) -> float:
    """
    Calculate the xy area of the periodic cell from an ASE Atoms object.

    :param atoms: ASE Atoms object.
    :return: Area of the xy plane.
    """
    cell = atoms.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    return area


def extract_total_wall_time_in_seconds(filename: str) -> float:
    """
    Reads a Quantum ESPRESSO output file, extracts the wall time, and converts it to seconds.
    Handles '35m22.08s', '35m 22.08s', '1h 3m', and '1h18m' formats.

    :param filename: Path to the Quantum ESPRESSO output file.
    :return: Total wall time in seconds, or None if not found.
    """
    try:
        with open(filename, 'r') as file:
            for line in file:
                if 'PWSCF' in line and 'WALL' in line:
                    parts = line.split()
                    wall_index = parts.index('WALL')
                    time_str = parts[wall_index - 1].replace('s', '')
                    hours, minutes, seconds = 0, 0, 0

                    if 'h' in time_str:
                        if 'm' in time_str:
                            hours, rest = time_str.split('h')
                            hours = int(hours)
                            minutes, seconds = (rest.split('m') + ['0'])[:2]
                        else:
                            hours = int(time_str.rstrip('h'))
                    elif 'm' in time_str:
                        minutes, seconds = (time_str.split('m') + ['0'])[:2]
                    else:
                        seconds = time_str

                    minutes = int(minutes) if minutes else 0
                    seconds = float(seconds) if seconds else 0

                    return hours * 3600 + minutes * 60 + seconds
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def add_allegro_number_array(ase_atom: Atoms, eps: float = .75, min_samples: int = 1) -> np.ndarray:
    """
    Assigns a number from 0 to 5 to each atom depending on its layer and position using DBSCAN clustering algorithm.

    :param ase_atom: ASE Atoms object with atom positions.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: A numpy array with assigned cluster numbers for each atom.
    """

    from sklearn.cluster import DBSCAN

    z_coords = ase_atom.positions[:, 2].reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(z_coords)
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    mean_positions = [z_coords[labels == label].mean() for label in unique_labels]
    sorted_indices = np.argsort(mean_positions)



    label_mapping = np.array([1, 2, 0, 4, 5, 3])
    # print(labels[:6],sorted_indices,sorted_indices[label_mapping])

    allegro_number_array = np.array([sorted_indices[label_mapping][label] if label != -1 else -1 for label in labels])

    return allegro_number_array

# Function to convert PhonopyAtoms to ASE Atoms object
def phonopy_atoms_to_ase(atoms_phonopy):
    """Convert PhonopyAtoms to Atoms."""
    try:
        from ase.atoms import Atoms
    except ImportError:
        raise ImportError("ASE python module was not found.")

    # Create and return a new Atoms object
    # with the data from the PhonopyAtoms object
    ase_atoms = Atoms(
        cell=atoms_phonopy.cell,
        scaled_positions=atoms_phonopy.scaled_positions,
        numbers=atoms_phonopy.numbers,
        pbc=True,
    )
    return ase_atoms


def gen_ase(cell, scaled_positions, numbers):
    """Convert PhonopyAtoms to Atoms."""
    try:
        from ase.atoms import Atoms
    except ImportError:
        raise ImportError("ASE python module was not found.")

    # Create and return a new Atoms object
    # with the data from the PhonopyAtoms object
    ase_atoms = Atoms(
        cell=cell,
        scaled_positions=scaled_positions,
        numbers=numbers,
        pbc=True,
    )
    return ase_atoms


# Function to convert ASE Atoms to PhonopyAtoms object
def ase_to_phonopy_atoms(ase_atoms):
    try:
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError("Phonopy python module was not found.")
    # Create and return a new PhonopyAtoms object
    # with the data from the ASE Atoms object
    phonopy_atoms = PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        scaled_positions=ase_atoms.get_scaled_positions(),
        cell=ase_atoms.cell,
    )

    return phonopy_atoms


# Function to generate qpoints along the BZ path given high-symmetry qpoints
def qpoints_Band_paths(HiSym_Qpoints, Nq_path):
    """Generate qpoints along the BZ path based on high-symmetry qpoints."""
    (Nq, DIM) = np.shape(HiSym_Qpoints)
    bands = []
    if Nq>=1:
        for iq in np.arange(0,Nq-1):
            qstart=HiSym_Qpoints[iq]
            qend=HiSym_Qpoints[iq+1]
            band=[]
            for i in np.arange(0,Nq_path+1):
                band.append(np.array(qstart)+(np.array(qend)-np.array(qstart))/Nq_path*i)
            bands.append(band)
    if Nq==1:
        bands.append(HiSym_Qpoints)
    return bands


def replace_line_starting_with(file_name, start_string, new_line):
    """
    Replaces lines in a file that start with a specified string with a new line.

    :param file_name: The path to the file
    :param start_string: The beginning string of the line to be replaced
    :param new_line: The new line that will replace the old line
    """
    # Read the file contents
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Replace lines starting with the specified string
    lines = [new_line+"\n" if line.startswith(start_string) else line for line in lines]

    # Write back the modified contents
    with open(file_name, 'w') as file:
        file.writelines(lines)

# Example usage
# replace_line_starting_with('path_to_your_file.txt', 'start_of_old_line', 'new_line_content')


# Example usage
# replace_line('path_to_your_file.txt', 'old_line_content', 'new_line_content')

def rotate_to_x_axis(atoms):
    """Rotate the structure so that the first lattice vector lies along the x-axis."""
    v1 = atoms.get_cell()[0]  # First lattice vector

    # Normalize the first lattice vector
    v1_norm = v1 / np.linalg.norm(v1)

    # Calculate the angle between v1 and the x-axis
    angle = np.arccos(np.clip(np.dot(v1_norm, [1, 0, 0]), -1.0, 1.0))

    # Axis for rotation: cross product of v1 and x-axis
    axis = np.cross(v1_norm, [1, 0, 0])

    # Check if rotation is necessary
    if np.linalg.norm(axis) > 1e-5 and not np.isnan(angle):
        # Rotate around the axis by the calculated angle
        atoms.rotate(v=axis, a=np.degrees(angle), rotate_cell=True)

    return atoms


from ase.io import write
from ase.io.trajectory import Trajectory

def traj_to_xyz(traj_path, xyz_path):
    """
    Convert a trajectory file to an extended XYZ file.

    Parameters:
    traj_path (str): Path to the input trajectory (.traj) file.
    xyz_path (str): Path for the output extended XYZ (.xyz) file.
    """
    traj = Trajectory(traj_path)
    images = [frame for frame in traj]

    write(xyz_path, images, format="extxyz")

    # # Example usage:
    # traj_path = "./nlayer_BP_relax.traj"
    # xyz_path = "nlayer_BP_traj.xyz"
    # traj_to_xyz(traj_path, xyz_path)