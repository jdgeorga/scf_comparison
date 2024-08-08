import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

from moirecompare.histograms.stacking_configuration import StackingConfigurationHistogrammer
from moirecompare.histograms.wasserstein import periodic_distance_matrix, wasserstein_distance_periodic

# ========================= USER INPUTS =========================
# Input and output file paths
INPUT_FILE = "structure_files/MoS2_WSe2-1.42_deg-2937_atoms.xyz"
OUTPUT_PREFIX = "output_plots/MoS2_WSe2-relaxed"

# Relaxation settings
RELAXED_FILE = "structure_files/MoS2_WSe2-relaxed.traj.xyz" 

# Layer files
LAYER_1_FILE = "structure_files/MoS2.xyz"
LAYER_2_FILE = "structure_files/WSe2.xyz"

# Layer symbols (adjust based on your structure)
LAYER_SYMBOLS = [
    ["Mo", "S", "S"],
    ["W", "Se", "Se"]
]

# Grid size for configuration space
GRID_SIZE = 12

# Histogram generation method ('kernel' or 'optimized')
METHOD = 'optimized'

# Regularization parameter for optimized method
LAMBDA_REG = 1e-4

# SOAP parameters (adjust as needed)
SOAP_PARAMS = {
    'species': [1, 2, 3, 4, 5, 6],
    'r_cut': 3.5,
    'n_max': 6,
    'l_max': 6,
    'sigma': 0.05,
    'periodic': True
}

# ========================= HELPER FUNCTIONS =========================

def setup_layers(layer1_file, layer2_file):
    # Load layer structures
    # IMPORTANT: assign the atom types based on your structure
    layer_1 = read(layer1_file)
    layer_2 = read(layer2_file)

    layer_1.arrays['atom_types'] = np.array([0,2,1])
    layer_2.arrays['atom_types'] = np.array([3,5,4])

    return layer_1, layer_2

# ========================= MAIN ANALYSIS =========================

def main():
    # Load original structure
    original_structure = read(INPUT_FILE)
  
    # Load relaxed structure
    print("Using provided relaxed structure...")
    relaxed_structure = read(RELAXED_FILE, index = -1, format = 'extxyz')
    relaxed_structure.arrays['atom_types'] = original_structure.arrays['atom_types']

    # Setup layers
    layer_1, layer_2 = setup_layers(LAYER_1_FILE, LAYER_2_FILE)

    # Generate histograms
    original_histogrammer = StackingConfigurationHistogrammer(original_structure)
    relaxed_histogrammer = StackingConfigurationHistogrammer(relaxed_structure)

    # Condition to select data center atom.Choose metal atom type in bottom layer.
    data_center_atom_cond = original_structure.arrays['atom_types'] == 0

    print("Generating Original Structure Histograms...")

    histogram_original = original_histogrammer.generate_histogram(
        layer_1, layer_2, GRID_SIZE, SOAP_PARAMS, data_center_atom_cond,
        method=METHOD, lambda_reg=LAMBDA_REG
    )

    print("Generating Relaxed Structure Histograms...")
    histogram_relaxed = relaxed_histogrammer.generate_histogram(
        layer_1, layer_2, GRID_SIZE, SOAP_PARAMS, data_center_atom_cond,
        method=METHOD, lambda_reg=LAMBDA_REG
    )

    # Calculate Wasserstein distance
    print("Calculating Wasserstein distance...")
    cell = (layer_1.get_cell()[:2, :2].copy() + layer_2.get_cell()[:2, :2].copy())/2
    distance_array = periodic_distance_matrix(GRID_SIZE, cell)
    wasserstein_dist = wasserstein_distance_periodic(histogram_original, histogram_relaxed, distance_array)

    print(f"Wasserstein distance between original and relaxed structures: {wasserstein_dist}")

    # Visualize results
    plt.figure(figsize=(16, 4))  # Increased figure width to accommodate labels
    
    plt.subplot(131)
    im1 = plt.imshow(histogram_original)
    plt.title("Original Structure Histogram")
    cbar1 = plt.colorbar(im1)
    cbar1.set_label('Cell Count', rotation=270, labelpad=15)
    
    plt.subplot(132)
    im2 = plt.imshow(histogram_relaxed)
    plt.title("Relaxed Structure Histogram")
    cbar2 = plt.colorbar(im2)
    cbar2.set_label('Cell Count', rotation=270, labelpad=15)
    
    plt.subplot(133)
    im3 = plt.imshow(np.abs(histogram_original - histogram_relaxed))
    plt.title("Histogram Difference")
    cbar3 = plt.colorbar(im3)
    cbar3.set_label('Cell Count Difference', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_histograms.png", dpi=300, bbox_inches='tight')
    print(f"Histogram visualizations saved to {OUTPUT_PREFIX}_histograms.png")

    # Update the distance metric array plot as well
    plt.figure(figsize=(6, 5))
    im4 = plt.imshow(distance_array[:,0].reshape(GRID_SIZE, GRID_SIZE))
    cbar4 = plt.colorbar(im4)
    cbar4.set_label('Distance (Å)', rotation=270, labelpad=15)
    plt.title('Distance Metric Array')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_distance_array.png", dpi=300, bbox_inches='tight')
    print(f"Distance array visualization saved to {OUTPUT_PREFIX}_distance_array.png")


if __name__ == "__main__":
    main()
