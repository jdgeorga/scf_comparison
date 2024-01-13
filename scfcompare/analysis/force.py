
import matplotlib.pyplot as plt
import numpy as np

def plot_forces_z_against_reference(data, reference_structure, window_size=10):
    """
    Plots the windowed mean and standard deviation of the z component of force differences
    for all structures against a chosen reference structure.

    :param ecut_data: Dictionary containing force data for all structures
    :param reference_structure: Data of the reference structure in ecut_data
    :param window_size: Size of the window for calculating mean and std (default 10)
    """

    # Extracting reference structure forces
    ref_forces = reference_structure['forces']
    ref_directory = reference_structure['directory']

    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.title(f'Windowed Mean and Standard Deviation against {ref_directory} (Window Size: {window_size})')
    plt.xlabel('Atom Index')
    plt.ylabel('Z Component Force Difference')

    # Loop through all structures and plot against reference
    for structure, data in data.items():
        

        # Calculating z component difference
        z_diff = data['forces'][:, 2] - ref_forces[:, 2]

        # Calculating windowed mean and std
        mean = np.convolve(z_diff, np.ones(window_size)/window_size, mode='valid')
        std = np.sqrt(np.convolve(z_diff**2, np.ones(window_size)/window_size, mode='valid') - mean**2)

        # Setting up indices for plotting
        indices = np.arange(len(mean)) + window_size // 2

        # Plotting
        plt.plot(indices, mean, label=f'Mean - {structure}', lw = 5)
        plt.fill_between(indices, mean - std, mean + std, alpha=0.3)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'z_forces_{ref_directory.split("/")[-1]}.png')
    
def plot_percentage_force_z_difference_against_reference(data, reference_structure, window_size=10):
    """
    Plots the windowed mean and standard deviation of the percentage difference in the z component of force 
    for all structures against a chosen reference structure.

    :param data: Dictionary containing force data for all structures
    :param reference_structure: Data of the reference structure in ecut_data
    :param window_size: Size of the window for calculating mean and std (default 10)
    """

    # Extracting reference structure forces
    ref_forces_z = reference_structure['forces'][:, 2]
    ref_directory = reference_structure['directory']

    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.title(f'Windowed Mean and Standard Deviation of Percentage Difference in Z Component Force against {ref_directory} (Window Size: {window_size})')
    plt.xlabel('Atom Index')
    plt.ylabel('Percentage Difference in Z Component Force')

    # Loop through all structures and plot against reference
    for structure, structure_data in data.items():
        # Calculating z component difference
        z_diff = structure_data['forces'][:, 2] - ref_forces_z

        # Calculating percentage difference
        # Avoid division by zero: add a small constant (e.g., 1e-9) to ref_forces_z
        percent_diff = (z_diff / (ref_forces_z + 1e-9)) * 100

        # Calculating windowed mean and std
        mean_percent_diff = np.convolve(percent_diff, np.ones(window_size) / window_size, mode='valid')
        std_percent_diff = np.sqrt(np.convolve(percent_diff ** 2, np.ones(window_size) / window_size, mode='valid') - mean_percent_diff ** 2)

        # Setting up indices for plotting
        indices = np.arange(len(mean_percent_diff)) + window_size // 2

        # Plotting
        plt.plot(indices, mean_percent_diff, label=f'Percentage Mean - {structure}', lw=5)
        plt.fill_between(indices, mean_percent_diff - std_percent_diff, mean_percent_diff + std_percent_diff, alpha=0.3)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-100,100)
    plt.savefig(f'perc_z_forces_{ref_directory.split("/")[-1]}.png')


# Example usage:
# plot_percentage_force_z_difference_against_reference(ecut_data, ecut_data['./1-ecut/9-ecut120'], window_size=10)


# Example usage:
# plot_forces_z_against_reference(ecut_data, ecut_data['./1-ecut/9-ecut120'], window_size=50)
# plot_percentage_force_z_difference_against_reference(ecut_data, ecut_data['./1-ecut/9-ecut120'], window_size=50)

# plot_forces_z_against_reference(zbox_data, zbox_data['./3-zbox/9b-z26'], window_size=50)
# plot_percentage_force_z_difference_against_reference(zbox_data, zbox_data['./3-zbox/9b-z26'], window_size=50)