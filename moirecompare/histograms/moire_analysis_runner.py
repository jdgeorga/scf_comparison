import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

from moirecompare.histograms.lammps_relax import lammps_relax
from moirecompare.histograms.stacking_configuration import StackingConfigurationHistogrammer
from moirecompare.histograms.wasserstein import periodic_distance_matrix, wasserstein_distance_periodic

# ========================= USER INPUTS =========================
# Input and output file paths
INPUT_FILE = "path/to/your/input_structure.xyz"
OUTPUT_PREFIX = "path/to/your/output_prefix"

# Relaxation settings
PERFORM_RELAXATION = True  # Set to False if using pre-relaxed structure
RELAXED_FILE = "path/to/your/relaxed_structure.xyz"  # Only used if PERFORM_RELAXATION is False

# Layer symbols (adjust based on your structure)
LAYER_SYMBOLS = [
    ["Mo", "S", "S"],
    ["W", "Se", "Se"]
]

# Grid size for configuration space
GRID_SIZE = 18

# Histogram generation method ('kernel' or 'optimized')
METHOD = 'optimized'

# Regularization parameter for optimized method
LAMBDA_REG = 0.1

# SOAP parameters (adjust as needed)
SOAP_PARAMS = {
    'species': [1, 2, 3, 4, 5, 6],
    'r_cut': 5.0,
    'n_max': 6,
    'l_max': 6,
    'sigma': 0.05,
    'periodic': True
}

# ========================= HELPER FUNCTIONS =========================

def setup_layers(structure):
    # TODO: Implement this function based on your specific structure
    # This is a placeholder. You need to split your structure into two layers.
    layer_1 = structure.copy()
    layer_2 = structure.copy()
    return layer_1, layer_2

# ========================= MAIN ANALYSIS =========================

def main():
    # Load original structure
    original_structure = read(INPUT_FILE)
    
    # Perform LAMMPS relaxation or load relaxed structure
    if PERFORM_RELAXATION:
        print("Performing LAMMPS relaxation...")
        relaxed_structure = lammps_relax(INPUT_FILE, OUTPUT_PREFIX, LAYER_SYMBOLS)
    else:
        print("Using provided relaxed structure...")
        relaxed_structure = read(RELAXED_FILE)

    # Setup layers
    layer_1, layer_2 = setup_layers(original_structure)

    # Generate histograms
    print("Generating histograms...")
    original_histogrammer = StackingConfigurationHistogrammer(original_structure)
    relaxed_histogrammer = StackingConfigurationHistogrammer(relaxed_structure)

    # TODO: Implement this condition based on your specific structure
    data_center_atom_cond = np.ones(len(original_structure), dtype=bool)

    histogram_original = original_histogrammer.generate_histogram(
        layer_1, layer_2, GRID_SIZE, SOAP_PARAMS, data_center_atom_cond,
        method=METHOD, lambda_reg=LAMBDA_REG
    )
    
    histogram_relaxed = relaxed_histogrammer.generate_histogram(
        layer_1, layer_2, GRID_SIZE, SOAP_PARAMS, data_center_atom_cond,
        method=METHOD, lambda_reg=LAMBDA_REG
    )

    # Calculate Wasserstein distance
    print("Calculating Wasserstein distance...")
    cell = original_structure.get_cell()[:2, :2]
    distance_array = periodic_distance_matrix(GRID_SIZE, cell)
    wasserstein_dist = wasserstein_distance_periodic(histogram_original, histogram_relaxed, distance_array)

    print(f"Wasserstein distance between original and relaxed structures: {wasserstein_dist}")

    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(histogram_original)
    plt.title("Original Structure Histogram")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(histogram_relaxed)
    plt.title("Relaxed Structure Histogram")
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(np.abs(histogram_original - histogram_relaxed))
    plt.title("Histogram Difference")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_histograms.png")
    print(f"Histogram visualizations saved to {OUTPUT_PREFIX}_histograms.png")

if __name__ == "__main__":
    main()
