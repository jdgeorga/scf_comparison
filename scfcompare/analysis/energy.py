import matplotlib.pyplot as plt

def plot_key_vs_convergence_values(data, ref_data, key):
    """
    Plot a specified key (e.g., total energy) against different convergence values for each directory in the data.

    :param data: Dictionary containing the data for all structures.
    :param key: Key for the value to plot in the data dictionary.
    """

    reference_value = ref_data[key]
    print(reference_value)

    for directory, info in data.items():
        conv_values = info['conv_values']
        values = info[key]
        plt.scatter(conv_values, values - reference_value, label=directory)

    plt.xlabel('Convergence Values')
    plt.ylabel(key.capitalize().replace('_', ' '))
    plt.title(f'{key.capitalize().replace("_", " ")} vs Convergence Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'diff_{key.capitalize().replace("_", " ")}.png')


def plot_percentage_difference_vs_convergence_values(data, ref_data, key):
    """
    Plot the percentage difference of a specified key (e.g., total energy) against different convergence values for each directory in the data.

    :param data: Dictionary containing the data for all structures.
    :param ref_data: Dictionary containing the reference data for comparison.
    :param key: Key for the value to plot in the data dictionary.
    """

    reference_value = ref_data[key]

    for directory, info in data.items():
        conv_values = info['conv_values']
        values = info[key]

        # Calculating the percentage difference
        # Avoid division by zero: add a small constant (e.g., 1e-12) to reference_value
        percentage_difference = (-(values - reference_value) / (reference_value + 1e-12)) * 100

        plt.scatter(conv_values, percentage_difference, label=directory)

    plt.xlabel('Convergence Values')
    plt.ylabel(f'Percentage Difference in {key.capitalize().replace("_", " ")}')
    plt.title(f'Percentage Difference in {key.capitalize().replace("_", " ")} vs Convergence Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'perc_diff_{key.capitalize().replace("_", " ")}.png')

# Example usage:
# plot_key_vs_convergence_values(ecut_data, ecut_data['./1-ecut/9-ecut120'], 'total_energy')
# plot_percentage_difference_vs_convergence_values(ecut_data, ecut_data['./1-ecut/9-ecut120'], 'total_energy')

# plot_key_vs_convergence_values(zbox_data, zbox_data['./3-zbox/9b-z26'], 'total_energy')
# plot_percentage_difference_vs_convergence_values(zbox_data, zbox_data['./3-zbox/9b-z26'], 'total_energy')

# plot_key_vs_convergence_values(econv_data, econv_data['./4-econv/3-e-10'], 'total_energy')
# plot_percentage_difference_vs_convergence_values(econv_data, econv_data['./4-econv/3-e-10'], 'total_energy')

# plot_key_vs_convergence_values(kpt_conv_data, kpt_conv_data['./6-kpt/3-k_3'], 'total_energy')
# plot_percentage_difference_vs_convergence_values(kpt_conv_data, kpt_conv_data['./6-kpt/3-k_3'], 'total_energy')