import numpy as np
from moirecompare.global_metrics.moire_pattern import MoirePatternAnalyzer
from ase.io import read
import matplotlib.pyplot as plt
import matplotlib as mpl

def extract_histograms(unrelaxed_structure, relaxed_structure, atom_types_array, layer_1, layer_2, N, soap_params):
    """
    Extract histograms of stacking configuration spaces for both unrelaxed and relaxed structures.
    
    Parameters:
    - unrelaxed_structure (Atoms): The unrelaxed atomic structure.
    - relaxed_structure (Atoms): The relaxed atomic structure.
    - atom_types_array (ndarray): Array of atom types.
    - layer_1 (Atoms): The first layer structure.
    - layer_2 (Atoms): The second layer structure.
    - N (int): Number of steps for translation in configuration space.
    - soap_params (dict): Parameters for the SOAP descriptor.
    
    Returns:
    - histogram (list of ndarray): Histograms of stacking configurations for unrelaxed and relaxed structures.
    """
    # Assign atom types and standardize positions and orientation for unrelaxed structure
    unrelaxed_structure.arrays['atom_types'] = atom_types_array
    unrelaxed_structure.positions -= unrelaxed_structure.positions[0]
    unrelaxed_structure.rotate(-np.arctan(unrelaxed_structure.cell[0, 1] / unrelaxed_structure.cell[0, 0]) * 180 / np.pi, 'z', rotate_cell=True)
    unrelaxed_structure.wrap()

    # Assign atom types and standardize positions and orientation for relaxed structure
    relaxed_structure.arrays['atom_types'] = atom_types_array
    relaxed_structure.positions -= relaxed_structure.positions[0]
    relaxed_structure.rotate(-np.arctan(relaxed_structure.cell[0, 1] / relaxed_structure.cell[0, 0]) * 180 / np.pi, 'z', rotate_cell=True)
    relaxed_structure.wrap()

    # Condition to select data center atom
    data_center_atom_cond = unrelaxed_structure.arrays['atom_types'] == 0

    # Instantiate MoirePatternAnalyzer and generate histogram
    analyzer = MoirePatternAnalyzer(unrelaxed_structure, relaxed_structure)
    histogram = analyzer.stacking_configuration_space_invariant_scale(layer_1,
                                                                      layer_2,
                                                                      N,
                                                                      soap_params,
                                                                      data_center_atom_cond,
                                                                      offset = 0.0, # Offset in Angstroms to break symmetries
                                                                      scale_thresh=0.01 # Threshold above which to keep histogram count
                                                                      )

    return histogram

if __name__ == "__main__":
    # Read input files for layer structures
    layer_1 = read("MoS2_monolayer.xyz")
    layer_2 = layer_1.copy()
    layer_1.positions -= layer_1.positions[0]
    layer_2.positions -= layer_2.positions[0]

    # Assign atom types to layers
    layer_1.arrays['atom_types'] = np.array([0, 2, 1], dtype=int)
    layer_2.arrays['atom_types'] = np.array([3, 5, 4], dtype=int)

    # Read unrelaxed and relaxed structures
    unrelaxed_structure = read("1D_25cells_0deg.xyz", index=0, format='extxyz')
    atom_types_array = unrelaxed_structure.arrays['atom_types']
    relaxed_structure = read("1D_MoS2_0deg_relax_25cells_FIRE_nlayer_lammps_traj.xyz", index=-1, format='extxyz')

    # Define parameters
    N = 12
    soap_params = {
        'species': [1, 2, 3, 4, 5, 6],
        'r_cut': 3.5,
        'n_max': 6,
        'l_max': 6,
        'sigma': 0.1,
        'periodic': True
    }

    # Extract histograms
    histograms = extract_histograms(unrelaxed_structure,
                                    relaxed_structure,
                                    atom_types_array,
                                    layer_1,
                                    layer_2,
                                    N,
                                    soap_params)

    # Plot the histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Define normalization
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max([np.max(histograms[0]), np.max(histograms[1])]))

    # Function to set custom ticks
    def set_custom_ticks(ax, histogram):
        num_ticks = histogram.shape[0]
        ticks = np.arange(num_ticks)
        tick_labels = ticks / num_ticks
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{label:.2f}" for label in tick_labels], rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{label:.2f}" for label in tick_labels])

    # Plot unrelaxed structure histogram
    im1 = axs[0].imshow(histograms[0], cmap='viridis', norm=norm)
    axs[0].set_title("Unrelaxed Structure")
    axs[0].set_xlabel("$δa_1$ in Configuration Space")
    axs[0].set_ylabel("$δa_2$ in Configuration Space")
    fig.colorbar(im1, ax=axs[0], label='Normalized Count', shrink=0.7)
    set_custom_ticks(axs[0], histograms[0])

    # Plot relaxed structure histogram
    im2 = axs[1].imshow(histograms[1], cmap='viridis', norm=norm)
    axs[1].set_title("Relaxed Structure")
    axs[1].set_xlabel("$δa_1$ in Configuration Space")
    axs[1].set_ylabel("$δa_2$ in Configuration Space")
    fig.colorbar(im2, ax=axs[1], label='Normalized Cell Count', shrink=0.7)
    set_custom_ticks(axs[1], histograms[1])

    # Rotate x-axis tick labels
    for ax in axs:
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("histograms.png", dpi=300, bbox_inches="tight")
    plt.show()