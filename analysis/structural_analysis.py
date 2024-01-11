
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def compute_prdf(positions, atom_types, pair_types, max_distance, bin_size, volume):
    """
    Compute the PRDF for specific pairs of atom types within a given structure.

    :param positions: Numpy array of atomic positions.
    :param atom_types: Numpy array of atomic types.
    :param pair_types: List of tuples for atom pairs to compute PRDF.
    :param max_distance: Maximum distance to consider for the PRDF.
    :param bin_size: Size of the bins for the histogram.
    :param volume: Total volume of the supercell.
    """
    prdf_data = {}
    
    for pair_type in pair_types:
        type_a, type_b = pair_type
        pos_a = positions[atom_types == type_a]
        pos_b = positions[atom_types == type_b]
        N_b = len(pos_b)

        # Initialize the PRDF array
        prdf = np.zeros(int(max_distance / bin_size))
        bin_edges = np.linspace(0, max_distance, len(prdf) + 1)

        for a in pos_a:
            # Compute distances from atom a to all atoms of type b
            distances = cdist([a], pos_b, 'euclidean').flatten()
            # Count number of atoms in each bin
            bin_counts = np.histogram(distances, bins=bin_edges)[0]
            prdf += bin_counts

        # Normalize by the shell volume and total number of b atoms, and scale by the total volume
        shell_volume = 4 * np.pi * (bin_edges[1:] ** 2) * bin_size
        prdf = prdf * volume / (N_b * shell_volume)

        # Store the computed PRDF
        prdf_data[pair_type] = (bin_edges[:-1], prdf)

    return prdf_data

def plot_prdf(data, pair_types, max_distance, bin_size):
    """
    Plot PRDFs for different structures with atom type information.

    :param data: Dictionary containing structures' data with atom types.
    :param pair_types: List of tuples for atom pairs to compute PRDF.
    :param max_distance: Maximum distance to consider for the PRDF.
    :param bin_size: Size of the bins for the histogram.
    """
    plt.figure(figsize=(12, 8))

    for structure_name, structure_data in data.items():
        positions = structure_data['positions']
        atom_types = structure_data['atom_types']
        volume = structure_data['volume']
        prdf_data = compute_prdf(positions, atom_types, pair_types, max_distance, bin_size, volume)
        for pair_type, (distances, prdf_values) in prdf_data.items():
            plt.plot(distances, prdf_values, label=f'{structure_name} - {pair_type}')

    plt.xlabel('Distance (r)')
    plt.ylabel('PRDF g(r)')
    plt.title('Partial Radial Distribution Functions')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_2d_prdf(positions, atom_types, pair_types, max_distance, bin_size, area):
    """
    Compute the 2D PRDF for specific pairs of atom types within a given structure.

    :param positions: Numpy array of atomic positions.
    :param atom_types: Numpy array of atomic types.
    :param pair_types: List of tuples for atom pairs to compute PRDF.
    :param max_distance: Maximum distance to consider for the PRDF.
    :param bin_size: Size of the bins for the histogram.
    :param area: Total area of the 2D layer.
    """
    prdf_data = {}
    
    for pair_type in pair_types:
        type_a, type_b = pair_type
        pos_a = positions[atom_types == type_a]
        pos_b = positions[atom_types == type_b]
        N_b = len(pos_b)

        # Initialize the PRDF array
        prdf = np.zeros(int(max_distance / bin_size))
        bin_edges = np.linspace(0, max_distance, len(prdf) + 1)

        for a in pos_a:
            # Compute distances from atom a to all atoms of type b
            distances = cdist([a], pos_b, 'euclidean').flatten()
            # Count number of atoms in each bin
            bin_counts = np.histogram(distances, bins=bin_edges)[0]
            prdf += bin_counts

        # Normalize by the ring area and total number of b atoms, and scale by the total area
        ring_area = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        prdf = prdf * area / (N_b * ring_area)

        # Store the computed PRDF
        prdf_data[pair_type] = (bin_edges[:-1], prdf)

    return prdf_data

def plot_2D_prdf(data, pair_types, max_distance, bin_size, xlim_range):
    """
    Plot 2D PRDFs for different structures with atom type information.

    :param data: Dictionary containing structures' data with atom types.
    :param pair_types: List of tuples for atom pairs to compute 2D PRDF.
    :param max_distance: Maximum distance to consider for the 2D PRDF.
    :param bin_size: Size of the bins for the histogram.
    :param xlim_range: Range for the x-axis limits.
    """
    plt.figure(figsize=(12, 8))
    
    for structure_name, structure_data in data.items():
        positions = structure_data['positions']
        atom_types = structure_data['atom_types']
        area = structure_data['area']
        prdf_data = compute_2d_prdf(positions, atom_types, pair_types, max_distance, bin_size, area)
        for pair_type, (distances, prdf_values) in prdf_data.items():
            plt.plot(distances, prdf_values, label=f'{structure_name} - {pair_type}')

    plt.xlabel('Distance (r)')
    plt.ylabel('PRDF g(r)')
    plt.title('2D Partial Radial Distribution Functions')
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim_range[0], xlim_range[1])
    plt.show()

def compute_difference_prdf(ref_prdf_data, comp_prdf_data, pair_type):
    """
    Compute the difference in 2D PRDF between a reference structure and a comparison structure for a given pair type.

    :param ref_prdf_data: PRDF data for the reference structure.
    :param comp_prdf_data: PRDF data for the comparison structure.
    :param pair_type: Tuple for atom pair to compute PRDF.
    :return: Tuple containing the distances and the difference in PRDF for the given pair type.
    """
    # Extract the PRDF data for the reference and comparison structures for the given pair type
    ref_distances, ref_prdf_values = ref_prdf_data[pair_type]
    comp_distances, comp_prdf_values = comp_prdf_data[pair_type]
    
    # Check that the distance bins are identical for the reference and comparison
    if not np.array_equal(ref_distances, comp_distances):
        raise ValueError("The distance bins for the reference and comparison PRDFs do not match.")
    
    # Calculate the difference in PRDF values
    prdf_difference = comp_prdf_values - ref_prdf_values
    
    return ref_distances, prdf_difference

def plot_difference_prdf(ref_data, comp_data, pair_types, max_distance, bin_size):
    """
    Plot the difference in PRDFs for a comparison structure against a reference structure for each pair type.

    :param ref_data: PRDF data for the reference structure.
    :param comp_data: Dictionary containing PRDF data for comparison structures.
    :param pair_types: List of tuples for atom pairs to compute PRDF.
    :param max_distance: Maximum distance to consider for the PRDF.
    :param bin_size: Size of the bins for the histogram.
    """
    # Compute PRDF for the reference structure
    ref_positions = ref_data['positions']
    ref_atom_types = ref_data['atom_types']
    ref_area = ref_data['area']
    ref_prdf_data = compute_2d_prdf(ref_positions, ref_atom_types, pair_types, max_distance, bin_size, ref_area)
    
    # Compute PRDF for each comparison structure
    for pair_type in pair_types:
        plt.figure(figsize=(12, 8))
        
        for structure_name, structure_data in comp_data.items():
            comp_positions = structure_data['positions']
            comp_atom_types = structure_data['atom_types']
            comp_area = structure_data['area']
            comp_prdf_data = compute_2d_prdf(comp_positions, comp_atom_types, [pair_type], max_distance, bin_size, comp_area)
            
            # Compute the difference in PRDF
            distances, prdf_difference = compute_difference_prdf(ref_prdf_data, comp_prdf_data, pair_type)
            
            # Plot the difference in PRDF for the given atom pair type
            plt.plot(distances, prdf_difference, label=f'Difference {structure_name} - {pair_type}')
        
        plt.xlabel('Distance (r)')
        plt.ylabel('Difference in PRDF g(r)')
        plt.title(f'Difference in PRDF for Atom Pair Type {pair_type}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
# data = extract_qe_data(directories, vals)
# volume = ...  # You should calculate or define the volume of your supercell here
# plot_prdf(ecut_data, [(0, 1)], max_distance=10, bin_size=0.1)
# plot_prdf(zbox_data, [(0, 1)], max_distance=10, bin_size=0.1)
# plot_difference_prdf(ref_conv_data['./7-most_converged/1-v1'], zbox_data, pair_types=[(0, 3), (1, 5)], max_distance=30, bin_size=0.05)

def calculate_z_displacements(positions, atom_types, type_a, type_b):
    """
    Calculate z-direction displacements between pairs of specified atom types.

    :param positions: Numpy array of atomic positions.
    :param atom_types: Numpy array of atomic types.
    :param type_a: The first atom type for displacement calculation.
    :param type_b: The second atom type for displacement calculation.
    :return: Array of z-direction displacements between each pair of specified atom types.
    """
    pos_a_z = positions[atom_types == type_a][:, 2]  # Z coordinates of type_a atoms
    pos_b_z = positions[atom_types == type_b][:, 2]  # Z coordinates of type_b atoms

    # Find the minimum z-direction displacement between each atom in pos_a and pos_b
    z_displacements = np.min(np.abs(pos_a_z[:, np.newaxis] - pos_b_z), axis=1)
    return z_displacements

def calculate_mean_z_displacement(data, ref_structure_key, type_a, type_b):
    """
    Calculate the mean displacement of z-direction distances between specific atom types,
    comparing a reference structure with all other structures in the dataset.

    :param data: Dictionary containing structure data.
    :param ref_structure_key: Key for the reference structure in the data dictionary.
    :param type_a: The first atom type for displacement comparison.
    :param type_b: The second atom type for displacement comparison.
    :return: Dictionary of mean displacements for each structure compared to the reference.
    """
    mean_displacements = {}

    # Calculate z-direction displacements in the reference structure
    ref_z_displacements = calculate_z_displacements(data[ref_structure_key]['positions'],
                                                    data[ref_structure_key]['atom_types'], 
                                                    type_a, type_b)

    # Compare with each other structure
    for key, structure in data.items():
        if key != ref_structure_key:
            comp_z_displacements = calculate_z_displacements(structure['positions'],
                                                             structure['atom_types'], 
                                                             type_a, type_b)

            # Calculate mean displacement
            mean_displacement = np.mean(np.abs(ref_z_displacements - comp_z_displacements))
            mean_displacements[key] = mean_displacement

    return mean_displacements

def plot_mean_z_displacements(data, ref_structure_key, atom_type_pairs):
    """
    Plot the mean z-direction displacements for each structure and each pair of atom types.

    :param data: Dictionary containing structure data.
    :param ref_structure_key: Key for the reference structure in the data dictionary.
    :param atom_type_pairs: List of tuples of atom type pairs to compare.
    """
    structure_names = [key for key in data.keys() if key != ref_structure_key]
    num_pairs = len(atom_type_pairs)

    # Prepare a scatter plot for mean displacements
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_pairs))

    for idx, (type_a, type_b) in enumerate(atom_type_pairs):
        mean_displacements = calculate_mean_z_displacement(data, ref_structure_key, type_a, type_b)
        ax.scatter(structure_names, [mean_displacements[name] for name in structure_names], 
                   label=f'Type {type_a} - Type {type_b}', color=colors[idx])

    ax.set_xlabel('Structure')
    ax.set_ylabel('Mean Z-Displacement')
    ax.set_title('Mean Z-Direction Displacements by Structure and Atom Type Pair')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()

def plot_absolute_z_distances(data, atom_type_pairs):
    """
    Plot the absolute z-direction distances for each structure and each pair of atom types.

    :param data: Dictionary containing structure data.
    :param atom_type_pairs: List of tuples of atom type pairs to compare.
    """
    structure_names = list(data.keys())
    num_pairs = len(atom_type_pairs)

    # Prepare a scatter plot for absolute z-direction distances
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_pairs))

    for idx, (type_a, type_b) in enumerate(atom_type_pairs):
        z_distances = {name: np.mean(calculate_z_displacements(structure['positions'], 
                                                              structure['atom_types'], 
                                                              type_a, type_b))
                       for name, structure in data.items()}
        ax.scatter(structure_names, [z_distances[name] for name in structure_names], 
                   label=f'Type {type_a} - Type {type_b}', color=colors[idx])

    ax.set_xlabel('Structure')
    ax.set_ylabel('Absolute Z-Distance')
    ax.set_title('Absolute Z-Direction Distances by Structure and Atom Type Pair')
    ax.legend()
    ax.axhline(y=0)
    plt.xticks(rotation=45)
    plt.show()

def calculate_percent_z_displacement(data, ref_structure_key, type_a, type_b):
    """
    Calculate the percent displacement of z-direction distances between specific atom types,
    comparing a reference structure with all other structures in the dataset.

    :param data: Dictionary containing structure data.
    :param ref_structure_key: Key for the reference structure in the data dictionary.
    :param type_a: The first atom type for displacement comparison.
    :param type_b: The second atom type for displacement comparison.
    :return: Dictionary of percent displacements for each structure compared to the reference.
    """
    percent_displacements = {}

    # Calculate z-direction displacements in the reference structure
    ref_z_displacements = calculate_z_displacements(data[ref_structure_key]['positions'],
                                                    data[ref_structure_key]['atom_types'], 
                                                    type_a, type_b)

    # Compare with each other structure
    for key, structure in data.items():
        if key != ref_structure_key:
            comp_z_displacements = calculate_z_displacements(structure['positions'],
                                                             structure['atom_types'], 
                                                             type_a, type_b)

            # Calculate percent displacement
            # Avoid division by zero: add a small constant (e.g., 1e-9) to ref_z_displacements
            percent_displacement = np.mean(np.abs(ref_z_displacements - comp_z_displacements) / (ref_z_displacements + 1e-9)) * 100
            percent_displacements[key] = percent_displacement

    return percent_displacements

def plot_percent_z_displacements(data, ref_structure_key, atom_type_pairs):
    """
    Plot the percent z-direction displacements for each structure and each pair of atom types.

    :param data: Dictionary containing structure data.
    :param ref_structure_key: Key for the reference structure in the data dictionary.
    :param atom_type_pairs: List of tuples of atom type pairs to compare.
    """
    structure_names = [key for key in data.keys() if key != ref_structure_key]
    num_pairs = len(atom_type_pairs)

    # Prepare a scatter plot for percent displacements
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_pairs))

    for idx, (type_a, type_b) in enumerate(atom_type_pairs):
        percent_displacements = calculate_percent_z_displacement(data, ref_structure_key, type_a, type_b)
        ax.scatter(structure_names, [percent_displacements[name] for name in structure_names], 
                   label=f'Type {type_a} - Type {type_b}', color=colors[idx])

    ax.set_xlabel('Structure')
    ax.set_ylabel('Percent Z-Displacement')
    ax.set_title('Percent Z-Direction Displacements by Structure and Atom Type Pair')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()



