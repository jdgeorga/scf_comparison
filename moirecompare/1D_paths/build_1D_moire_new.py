import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt
import pymoire as pm
from ase.build.tools import cut
import time
from ase.io import read, write
from ase.geometry import get_duplicate_atoms


def zero_xaxis(atoms: Atoms):
    cell = atoms.cell
    theta = np.arctan(cell[0,1]/cell[0,0])

    atoms.rotate(-theta*180/np.pi,'z',rotate_cell = True)

    return atoms

def generate_1D_moire(layer_1, layer_2, N, scaling_vector, delta_scalar, plot = True):

    n_b = N
    n_t = N + delta_scalar

    # Define translation vectors for bottom and top layers
    N_b_1 = np.array([n_b, 0, 0])
    N_b_2 = np.array([0, 1, 0])
    N_b_1_bar = N_b_1 + scaling_vector
    N_t_1 = np.array([n_t, 0, 0])
    N_t_2 = np.array([0, 1, 0])

    # Copy layers and adjust atom types for top layer
    bottom_layer: Atoms = layer_1.copy()[:]
    top_layer = layer_2.copy()[:]
    top_layer.arrays['atom_types'] += 3

    # Adjust cell dimensions and center layers
    bottom_layer.cell[2, 2] = 30
    top_layer.cell[2, 2] = 30
    bottom_layer.center(axis=2)
    top_layer.center(axis=2)

    # Get cell vectors
    A_b_1 = bottom_layer.cell[0]
    A_b_2 = bottom_layer.cell[1]
    A_t_1 = top_layer.cell[0]
    A_t_2 = top_layer.cell[1]

    # Calculate new cell for top layer
    M_1 = N_b_1_bar @ bottom_layer.cell
    M_set = np.zeros_like(bottom_layer.cell)
    M_set[0] = M_1
    M_set[1] = A_t_2
    M_set[2, 2] = 30.0

    # Calculate rotation angle and apply rotation
    theta = np.arctan(M_1[1] / M_1[0])
    print("Angle:", theta * 180 / np.pi)
    top_layer.rotate(theta * 180 / np.pi, 'z', rotate_cell=True)

    # Calculate strain and adjust cell
    d_1 = M_1 * 1 / (N_t_1 @ top_layer.cell)
    d_1 = np.nan_to_num(d_1, nan=1.0)
    strain = np.sum(d_1 - np.array([1, 1, 1]))
    print("Strain:", strain)
    top_layer_cell_bar = top_layer.cell.copy()
    top_layer_cell_bar[0] = d_1 * top_layer_cell_bar[0]
    top_layer.set_cell(top_layer_cell_bar, scale_atoms=True)

    # Cut the layers to form 1D structure
    bottom_layer = cut(bottom_layer, a=(int(N_b_1_bar[0]), int(N_b_1_bar[1]), int(N_b_1_bar[2])), b=(int(N_b_2[0]), int(N_b_2[1]), int(N_b_2[2])), c=(0, 0, 1))
    top_layer = cut(top_layer, a=N_t_1, b=N_t_2, c=(0, 0, 1))

    # Set cells to match and combine layers
    top_layer.set_cell(M_set, scale_atoms=False)
    bottom_layer.set_cell(M_set, scale_atoms=False)

    # Adjust positions to create bilayer structure
    displacement = (bottom_layer.positions[:, 2].max() - bottom_layer.cell[2, 2] / 2 + 1.5).copy()
    bottom_layer.positions[:, 2] -= displacement
    top_layer.positions[:, 2] += displacement

    # Combine layers and repeat structure
    combined_structure = bottom_layer.copy() + top_layer.copy()
    combined_structure.wrap()

    combined_structure = zero_xaxis(combined_structure)
    combined_structure.wrap()

    print(get_duplicate_atoms(combined_structure))

    if plot is True:
        combined_structure_plot = combined_structure.repeat([2, 30, 1])
        combined_structure_plot.wrap()
        # Plot and save the structure
        plt.figure(figsize=(20, 20))
        plt.scatter(combined_structure_plot.positions[combined_structure_plot.arrays['atom_types'] < 1, 0], combined_structure_plot.positions[combined_structure_plot.arrays['atom_types'] < 1, 1], s=20)
        plt.scatter(combined_structure_plot.positions[combined_structure_plot.arrays['atom_types'] == 3, 0], combined_structure_plot.positions[combined_structure_plot.arrays['atom_types'] == 3, 1], s=10)
        plt.axis('scaled')
        plt.savefig("1D_moire.png")

    return combined_structure, strain, theta

if __name__ == '__main__':

    # Load material data from pymoire
    materials_db_path = pm.materials.get_materials_db_path()
    layer_1: Atoms = pm.read_monolayer(materials_db_path / 'MoS2.cif')
    layer_2 = pm.read_monolayer(materials_db_path / 'MoS2.cif')

    # Center the positions
    layer_1.positions -= layer_1.positions[0]
    layer_2.positions -= layer_2.positions[0]

    # Define atom types for distinguishing between layers
    layer_1.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
    layer_2.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)

    # Initialize timer
    start_time = time.time()

    # Define the supercell scaling factor and number of repetitions
    scaling_vector = np.array([0, 1, 0]) * 2
    delta_scalar = -1
    N = 30

    theta_list = []
    strain_list = []
    n_atoms_list = []

    deltas = np.arange(-10, 10)
    scales = np.arange(10)

    deltas = [0,-1]
    scales = [1,2]
    Ns = [4,8]
    
    for i,j, n in zip(deltas,scales,Ns):
        delta_scalar = i
        scaling_vector = np.array([0, 1, 0]) * j
        atoms, strain, theta = generate_1D_moire(layer_1, layer_2, n, scaling_vector, delta_scalar, plot = True)
        theta_list.append(theta * 180 / np.pi)
        strain_list.append(strain)
        n_atoms_list.append(len(atoms))
        print("Number of atoms:", len(atoms[atoms.arrays['atom_types'] < 1]))
        print(i, j, theta, strain, len(atoms))
        
        write("test.cif", atoms, format = "cif")
        write("MoS2_1D_13.89deg_8atoms.xyz", atoms, format = "extxyz")

    plt.figure(figsize=(20, 20))
    plt.scatter(theta_list, strain_list, c = n_atoms_list)
    plt.savefig("theta_vs_strain.png", dpi=300, bbox_inches='tight')

    # Print elapsed time
    print("Elapsed time:", time.time() - start_time)
