import os
from scfcompare.io.file_operations import extract_qe_data, get_subdirectories
from scfcompare.analysis import energy
from scfcompare.analysis import force
from scfcompare.analysis import structural


ecut_path = '/scratch1/08526/jdgeorga/relax_test/1-ecut'
zbox_path = '/scratch1/08526/jdgeorga/relax_test/3-zbox'

ecut_ref = '9-ecut120'
zbox_ref = '9b-z26' 

ecut_dirs = get_subdirectories(ecut_path)
zbox_dirs = get_subdirectories(zbox_path)


ecut_vals = [40,50,60,70,75,80,90,100,110,120]
zbox_vals = [15,16,17,18,19, 20, 21, 22, 23, 25,26]

ecut_data = extract_qe_data(ecut_dirs, ecut_vals)
zbox_data = extract_qe_data(zbox_dirs, zbox_vals)

# # Energy analysis
# energy.plot_key_vs_convergence_values(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'], 'total_energy')
# energy.plot_percentage_difference_vs_convergence_values(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'], 'total_energy')

# energy.plot_key_vs_convergence_values(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'], 'total_energy')
# energy.plot_percentage_difference_vs_convergence_values(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'], 'total_energy')

# # # Force analysis

# force.plot_forces_z_against_reference(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'],window_size=50)
# force.plot_percentage_force_z_difference_against_reference(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'],window_size=50)

# force.plot_forces_z_against_reference(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'],window_size=50)
# force.plot_percentage_force_z_difference_against_reference(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'],window_size=50)

# # Structural analysis

# structural.plot_prdf(ecut_data,[(0,1)],max_distance= 20, bin_size= 0.01)
# # structural.plot_2D_prdf
# # structural.plot_mean_z_displacements
# # structural.plot_absolute_z_distances
# # structural.plot_percent_z_displacements

