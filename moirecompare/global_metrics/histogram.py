import numpy as np
from scipy.stats import wasserstein_distance_nd
import ot

import matplotlib.pyplot as plt

from moirecompare.global_metrics.moire_pattern import MoirePatternAnalyzer


def periodic_distance_matrix(size, cell):
    """
    Compute the pairwise distance matrix for a 2D periodic grid of given size
    in crystal coordinates defined by the cell matrix.
    
    Parameters:
    - size: int, the size of the square grid.
    - cell: 2x2 array-like, the 2D cell matrix of the lattice.
    
    Returns:
    - dist_matrix: 2D array, the pairwise distance matrix.
    """
    # Generate fractional coordinates for the grid points
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    fractional_coords = np.column_stack([x.ravel(), y.ravel()]) / size
    
    # Convert fractional coordinates to crystal coordinates using the cell matrix
    crystal_coords = fractional_coords @ cell
    
    # Initialize the distance matrix
    dist_matrix = np.zeros((size * size, size * size))
    
    # Compute the pairwise distances considering periodic boundary conditions
    for i, (x1, y1) in enumerate(crystal_coords):
        for j, (x2, y2) in enumerate(crystal_coords):
            dx = min(abs(x1 - x2), abs(x1 - x2 + cell[0,0]), abs(x1 - x2 - cell[0,0]))
            dy = min(abs(y1 - y2), abs(y1 - y2 + cell[1,1]), abs(y1 - y2 - cell[1,1]))
            dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
    
    return dist_matrix


def wasserstein_distance_periodic(histogram1, histogram2, cell):
    # Ensure the histograms are the same shape
    assert histogram1.shape == histogram2.shape, "Histograms must be of the same shape."
    
    # Get the size of the square matrix
    size = histogram1.shape[0]
    
    # Normalize histograms to ensure they sum to 1
    hist1 = histogram1 / np.sum(histogram1)
    hist2 = histogram2 / np.sum(histogram2)
    
    # Generate coordinates for the 2D grid
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    coordinates = np.column_stack([x.ravel(), y.ravel()])
    
    # Flatten the histograms for Wasserstein distance calculation
    hist1_flat = hist1.ravel()
    hist2_flat = hist2.ravel()
    
    # Compute the periodic distance matrix
    pairwise_distances = periodic_distance_matrix(size, cell)
    
    # Compute the Wasserstein distance using the ot library
    distance = ot.emd2(hist1_flat, hist2_flat, pairwise_distances)
    
    return distance


def compute_wasserstein_matrix(histogram_list, cell):
    print("Computing Wasserstein Matrix")

    n = len(histogram_list)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                w_dist = wasserstein_distance_periodic(histogram_list[i], histogram_list[j], cell)
                distance_matrix[i, j] = w_dist
    
    return distance_matrix


def plot_wasserstein_matrix(distance_matrix, histogram_label_list):
    print("Plotting Wasserstein Matrix")
    fig, ax = plt.subplots()
    cax = ax.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(cax)

    # Label the axes
    ax.set_xticks(np.arange(len(histogram_label_list)))
    ax.set_yticks(np.arange(len(histogram_label_list)))
    ax.set_xticklabels(histogram_label_list)
    ax.set_yticklabels(histogram_label_list)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(histogram_label_list)):
        for j in range(len(histogram_label_list)):
            text = ax.text(j, i, f"{distance_matrix[i, j]:.3f}", ha="center", va="center", color="w")

    ax.set_title("Wasserstein Distance Matrix")
    fig.tight_layout()
    plt.savefig("wasserstein_distance_matrix.png", dpi=300, bbox_inches="tight")

def extract_histogram(unrelaxed_structure,
                      relaxed_structure,
                      atom_types_array,
                      layer_1,
                      layer_2,
                      N,
                      soap_params):
    unrelaxed_structure.arrays['atom_types'] = atom_types_array
    unrelaxed_structure.positions -= unrelaxed_structure.positions[0]
    unrelaxed_structure.rotate(-np.arctan(unrelaxed_structure.cell[0,1]/unrelaxed_structure.cell[0,0]) * 180/np.pi, 'z', rotate_cell = True)
    unrelaxed_structure.wrap()

    relaxed_structure.arrays['atom_types'] = atom_types_array
    relaxed_structure.positions -= relaxed_structure.positions[0]
    relaxed_structure.rotate(-np.arctan(relaxed_structure.cell[0,1]/relaxed_structure.cell[0,0]) * 180/np.pi, 'z', rotate_cell = True)
    relaxed_structure.wrap()

    data_2D_mo_cond = unrelaxed_structure.arrays['atom_types'] == 0

    analyzer = MoirePatternAnalyzer(unrelaxed_structure, relaxed_structure)
    histogram = analyzer.stacking_configuration_space(layer_1, layer_2, N, soap_params, data_2D_mo_cond)

    return histogram

if __name__ == "__main__":

    import pymoire as pm
    import numpy as np
    from ase.io import read, write
    p = pm.materials.get_materials_db_path()

    # Load structures and initialize layers
    p = pm.materials.get_materials_db_path()
    layer_1 = pm.read_monolayer(p / 'MoS2.cif')
    layer_2 = pm.read_monolayer(p / 'MoS2.cif')
    layer_1.positions -= layer_1.positions[0]
    layer_2.positions -= layer_2.positions[0]
    layer_1.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
    layer_2.arrays['atom_types'] = np.array([3, 5, 4], dtype=int)
    

    # Initialize other parameters
    N = 12
    soap_params = {
        'species': [1, 2, 3, 4, 5, 6],
        'r_cut': 3.5,
        'n_max': 6,
        'l_max': 6,
        'sigma': 0.1,
        'periodic': True
    }

    unrelaxed_structure = read("1D_25cells_0deg.xyz",
                               index=0,
                               format='extxyz')
    atom_types_array = read("1D_25cells_0deg.xyz").arrays['atom_types']

    print("Reading Structures")
    relaxed_structure_list = [
        read("1D_MoS2_0deg_relax_25cells_FIRE_traj_f4e-3.xyz", index  = -1), # allegro
        read("1D_MoS2_0deg_relax_25cells_FIRE_nlayer_lammps_traj.xyz", index  = -1), # lammps
        read("1D_MoS2_from_ALLEGRO_relax.pwo", index=-1, format='espresso-out'), # allegroDFT
        read("1D_MoS2_from_LAMMPS_relax.pwo", index=-1, format='espresso-out') # lammpsDFT
    ]

    print("Compiling Histogram List")
    histogram_list = [extract_histogram(unrelaxed_structure,
                                        relaxed_structure_list[0],
                                        atom_types_array,
                                        layer_1,
                                        layer_2,
                                        N,
                                        soap_params) [0]]
    for relaxed_structure in relaxed_structure_list:

        histogram = extract_histogram(unrelaxed_structure,
                                      relaxed_structure,
                                      atom_types_array,
                                      layer_1,
                                      layer_2,
                                      N,
                                      soap_params)

        histogram_list.append(histogram[1])

    histogram_label_list = ["Initial Unrelax",
                            "Allegro Relax",
                            "SW+KC Relax",
                            "DFT Relax (w/ Allegro start)",
                            "DFT Relax (w/ SW+KC start)"]
    cell = layer_1.cell[:2, :2]

    distance_matrix = compute_wasserstein_matrix(histogram_list, cell)
    plot_wasserstein_matrix(distance_matrix, histogram_label_list)
