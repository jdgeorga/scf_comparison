
import matplotlib.pyplot as plt
import numpy as np

def plot_forces_z_against_reference(atoms, ref_atom, window_size=10):
    """
    Plots the windowed mean and standard deviation of the z component of force differences
    for all structures against a chosen reference structure.

    :param ecut_data: Dictionary containing force data for all structures
    :param reference_structure: Data of the reference structure in ecut_data
    :param window_size: Size of the window for calculating mean and std (default 10)
    """

    # Extracting reference structure forces
    ref_forces = ref_atom.get_forces()
    ref_directory = ref_atom.arrays['directory']

    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.title(f'Windowed Mean and Standard Deviation against {ref_directory} (Window Size: {window_size})')
    plt.xlabel('Atom Index')
    plt.ylabel('Z Component Force Difference')

    # Loop through all structures and plot against reference
    for structure, atom in atoms.items():
        
        # Calculating z component difference
        z_diff = atom.get_forces()[:, 2] - ref_forces[:, 2]

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
    
def plot_percentage_force_z_difference_against_reference(atoms, ref_atom, window_size=10):
    """
    Plots the windowed mean and standard deviation of the percentage difference in the z component of force 
    for all structures against a chosen reference structure.

    :param data: Dictionary containing force data for all structures
    :param reference_structure: Data of the reference structure in ecut_data
    :param window_size: Size of the window for calculating mean and std (default 10)
    """

    # Extracting reference structure forces
    ref_forces_z = ref_atom.get_forces()[:, 2]
    ref_directory = ref_atom.arrays['directory']

    # Plot setup
    plt.figure(figsize=(12, 8))
    plt.title(f'Windowed Mean and Standard Deviation of Percentage Difference in Z Component Force against {ref_directory} (Window Size: {window_size})')
    plt.xlabel('Atom Index')
    plt.ylabel('Percentage Difference in Z Component Force')

    # Loop through all structures and plot against reference
    for structure, atom in atoms.items():
        # Calculating z component difference
        z_diff = atom.get_forces()[:, 2] - ref_forces_z

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
    

def plot_mean_force_diff_by_atom_type(atoms, ref_atom, atom_types):
    """
    Plots the mean force difference in the z direction for individual atom types in subplots.

    :param atoms: Dictionary containing force data for all structures.
    :param ref_atom: Data of the reference structure.
    :param atom_types: List of atom types to consider.
    """

    # Extracting reference structure forces
    ref_forces_z = ref_atom.get_forces()[:, 2]

    # Number of subplots
    n_subplots = len(atom_types)
    fig, axes = plt.subplots(n_subplots, 1, 
                             figsize=(12, 12),
                             constrained_layout=True)
    
    if n_subplots == 1:
        axes = [axes]  # To handle single subplot case

    # Plotting each atom type in a separate subplot
    for i, atom_type in enumerate(atom_types):
        axes[i].set_title(f'Mean Z Force Difference for Atom Type: {atom_type}')
        axes[i].set_xlabel('Atom Index')
        axes[i].set_ylabel('Mean Z Force Difference')

        for structure, atom in atoms.items():
            # Filter forces by atom type
            type_mask = (atom.arrays['atom_types'] == atom_type)
            atom_forces_z = atom.get_forces()[:, 2][type_mask]
            ref_forces_z_filtered = ref_forces_z[type_mask]
            conv_values = atom.arrays['conv_values']

            if len(atom_forces_z) > 0 and len(ref_forces_z_filtered) > 0:
                # Calculate mean force difference
                mean_diff = np.mean(atom_forces_z - ref_forces_z_filtered)
                # Plotting
                axes[i].plot(conv_values, mean_diff, 'o', label=f'{structure} (Mean diff: {mean_diff:.2f})')
            else:
                # Handling cases where atom type is not present
                axes[i].plot([], [], 'o', label=f'{structure} (No data for this atom type)')

        # axes[i].legend()
        axes[i].grid(True)

    # plt.tight_layout()
    plt.savefig(f'mean_force_diff_by_atom_type_{structure.split("/")[-2]}.png')

# Example usage:
# atom_types = [1, 2]  # Replace with actual atomic numbers of interest
# plot_mean_force_diff_by_atom_type(ecut_data, ecut_data['./1-ecut/9-ecut120'], atom_types)

def plot_mean_percentage_force_diff_by_atom_type(atoms, ref_atom, atom_types):
    """
    Plots the mean percentage force difference in the z direction for individual atom types in subplots.

    :param atoms: Dictionary containing force data for all structures.
    :param ref_atom: Data of the reference structure.
    :param atom_types: List of atom types to consider.
    """

    # Extracting reference structure forces
    ref_forces_z = ref_atom.get_forces()[:, 2]

    # Number of subplots
    n_subplots = len(atom_types)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 12), constrained_layout=True)
    
    if n_subplots == 1:
        axes = [axes]  # To handle single subplot case

    # Plotting each atom type in a separate subplot
    for i, atom_type in enumerate(atom_types):
        axes[i].set_title(f'Mean Percentage Z Force Difference for Atom Type: {atom_type}')
        axes[i].set_xlabel('Atom Index')
        axes[i].set_ylabel('Mean Percentage Z Force Difference')

        for structure, atom in atoms.items():
            # Filter forces by atom type
            type_mask = (atom.arrays['atom_types'] == atom_type)
            atom_forces_z = atom.get_forces()[:, 2][type_mask]
            ref_forces_z_filtered = ref_forces_z[type_mask]
            conv_values = atom.arrays['conv_values']

            if len(atom_forces_z) > 0 and len(ref_forces_z_filtered) > 0:
                # Avoid division by zero: add a small constant to ref_forces_z_filtered
                safe_ref_forces_z = ref_forces_z_filtered + 1e-9

                # Calculate mean percentage difference
                mean_percentage_diff = np.mean((atom_forces_z - ref_forces_z_filtered) / safe_ref_forces_z * 100)
                # Plotting
                axes[i].plot(conv_values, mean_percentage_diff, 'o', label=f'{structure} (Mean % diff: {mean_percentage_diff:.2f})')
            else:
                # Handling cases where atom type is not present
                axes[i].plot([], [], 'o', label=f'{structure} (No data for this atom type)')

        # axes[i].legend()
        axes[i].grid(True)

    # plt.tight_layout()
    plt.savefig(f'mean_percentage_force_diff_by_atom_type_{structure.split("/")[-2]}.png')
