import numpy as np
import ot

def periodic_distance_matrix(size, cell):
    """
    Compute the pairwise distance matrix for a 2D periodic grid of given size
    in crystal coordinates defined by the cell matrix.
    
    Args:
    size (int): The size of the square grid.
    cell (np.array): 2x2 array, the 2D cell matrix of the lattice.
    
    Returns:
    np.array: The pairwise distance matrix.
    """
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    fractional_coords = np.column_stack([x.ravel(), y.ravel()]) / size
    
    cartesian_coords = fractional_coords @ cell
    
    dist_matrix = np.full((size * size, size * size), np.inf)
    
    translation_vectors = [
        np.array([0, 0]),
        cell[0], cell[1], cell[0] + cell[1],
        -cell[0], -cell[1], -cell[0] - cell[1],
        cell[0] - cell[1], -cell[0] + cell[1]
    ]
    
    for translation in translation_vectors:
        translated_coords = cartesian_coords + translation
        for i in range(size * size):
            for j in range(size * size):
                dist = np.linalg.norm(cartesian_coords[i] - translated_coords[j])
                if dist < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist
    
    return dist_matrix

def is_symmetric(matrix, rtol=1e-05, atol=1e-08):
    """
    Check if a matrix is symmetric.
    
    Args:
    matrix (np.array): The matrix to check for symmetry.
    rtol (float): The relative tolerance parameter (default is 1e-05).
    atol (float): The absolute tolerance parameter (default is 1e-08).
    
    Returns:
    bool: True if the matrix is symmetric, False otherwise.
    """
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def wasserstein_distance_periodic(histogram1, histogram2, pairwise_distances):
    """
    Calculate the Wasserstein distance between two histograms considering periodic boundary conditions.
    
    Args:
    histogram1 (np.array): First histogram.
    histogram2 (np.array): Second histogram.
    pairwise_distances (np.array): Pairwise distance matrix.
    
    Returns:
    float: The Wasserstein distance between the two histograms.
    """
    assert histogram1.shape == histogram2.shape, "Histograms must be of the same shape."
    
    size = histogram1.shape[0]
    
    hist1 = histogram1 / np.sum(histogram1)
    hist2 = histogram2 / np.sum(histogram2)

    hist1_flat = hist1.ravel()
    hist2_flat = hist2.ravel()
    
    distance = ot.emd2(hist1_flat, hist2_flat, pairwise_distances)
    
    return distance

def compute_wasserstein_distances(ref_list, histogram_list, distance_array):
    """
    Compute Wasserstein distances between reference histograms and a list of histograms.
    
    Args:
    ref_list (list): List of reference histograms.
    histogram_list (list): List of histograms to compare against the references.
    distance_array (np.array): Pairwise distance matrix.
    
    Returns:
    np.array: Array of Wasserstein distances.
    """
    n = len(ref_list)
    m = len(histogram_list)

    distances = np.zeros(m)
    
    if m == n:
        for i in range(m):
            distances[i] = wasserstein_distance_periodic(ref_list[i], histogram_list[i], distance_array)
    elif n == 1:
        for i in range(m):
            distances[i] = wasserstein_distance_periodic(ref_list[0], histogram_list[i], distance_array)
    else: 
        raise ValueError("Reference and Histogram list must be of the same length or reference list must have only one element")

    return distances

def main():
    # Example usage
    N = 10
    cell = np.array([[1.0, 0.0], [0.5, 0.866]])  # Example hexagonal cell
    
    distance_array = periodic_distance_matrix(N, cell)
    
    # Check if distance_array is symmetric
    if is_symmetric(distance_array):
        print("The distance array is symmetric.")
    else:
        print("Warning: The distance array is not symmetric!")
    
    # Create example histograms
    hist1 = np.random.rand(N, N)
    hist2 = np.random.rand(N, N)
    
    # Compute Wasserstein distance
    distance = wasserstein_distance_periodic(hist1, hist2, distance_array)
    print(f"Wasserstein distance between histograms: {distance}")
    
    # Compute distances for multiple histograms
    ref_hist = np.random.rand(N, N)
    hist_list = [np.random.rand(N, N) for _ in range(5)]
    
    distances = compute_wasserstein_distances([ref_hist], hist_list, distance_array)
    print("Wasserstein distances for multiple histograms:")
    print(distances)

if __name__ == "__main__":
    main()
