import os
from scfcompare.io.file_operations import get_atoms_list_from_QE, get_subdirectories
from scfcompare.analysis import energy
from scfcompare.analysis import force
from scfcompare.analysis import structural


ecut_path = '/scratch1/08526/jdgeorga/relax_test/1-ecut'
zbox_path = '/scratch1/08526/jdgeorga/relax_test/3-zbox'

ecut_ref = '9-ecut120'
zbox_ref = '9a-z25' 

ecut_dirs = get_subdirectories(ecut_path)
zbox_dirs = get_subdirectories(zbox_path)


ecut_vals = [40,50,60,70,75,80,90,100,110,120]
zbox_vals = [15,16,17,18,19, 20, 21, 22, 23, 25,26]

ecut_data = get_atoms_list_from_QE(ecut_dirs, ecut_vals)
zbox_data = get_atoms_list_from_QE(zbox_dirs, zbox_vals)
zbox_data.pop(f'{zbox_path}/1-z15')
zbox_data.pop(f'{zbox_path}/9b-z26')

ecut_ref_atom = ecut_data[f'{ecut_path}/{ecut_ref}']
zbox_ref_atom = zbox_data[f'{zbox_path}/{zbox_ref}']

# # Energy analysis
# energy.plot_energy_vs_convergence_values(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'])
# energy.plot_energy_percentage_difference_vs_convergence_values(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'])

# energy.plot_energy_vs_convergence_values(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'])
# energy.plot_energy_percentage_difference_vs_convergence_values(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'])

# # # Force analysis

#force.plot_forces_z_against_reference(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'],window_size=50)
#force.plot_percentage_force_z_difference_against_reference(ecut_data, ecut_data[f'{ecut_path}/{ecut_ref}'],window_size=50)

# force.plot_forces_z_against_reference(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'],window_size=50)
# force.plot_percentage_force_z_difference_against_reference(zbox_data, zbox_data[f'{zbox_path}/{zbox_ref}'],window_size=50)

force.plot_mean_force_diff_by_atom_type(ecut_data,ecut_ref_atom,atom_types=[0,1,2,3,4,5])
force.plot_mean_percentage_force_diff_by_atom_type(ecut_data,ecut_ref_atom,atom_types=[0,1,2,3,4,5])

force.plot_mean_force_diff_by_atom_type(zbox_data,zbox_ref_atom,atom_types=[0,1,2,3,4,5])
force.plot_mean_percentage_force_diff_by_atom_type(zbox_data,zbox_ref_atom,atom_types=[0,1,2,3,4,5])



# # Structural analysis

# structural.plot_prdf(ecut_data,[(0,1)],max_distance= 20, bin_size= 0.01)
# structural.plot_2D_prdf(ecut_data,[(0,1)],max_distance = 20, bin_size = 0.01)
# structural.plot_mean_z_displacements(ecut_data,ecut_data[f'{ecut_path}/{ecut_ref}'],atom_type_pairs=[(0,1)])
# structural.plot_absolute_z_distances(ecut_data,atom_type_pairs=[(0,1)])
# structural.plot_percent_z_displacements(ecut_data,ecut_data[f'{ecut_path}/{ecut_ref}'],atom_type_pairs=[(0,1)])

# structural.plot_prdf(zbox_data,[(0,1)],max_distance= 20, bin_size= 0.01)
# structural.plot_2D_prdf(zbox_data,[(0,1)],max_distance = 20, bin_size = 0.01)
# structural.plot_mean_z_displacements(zbox_data,zbox_data[f'{zbox_path}/{zbox_ref}'],atom_type_pairs=[(0,1)])
# structural.plot_absolute_z_distances(zbox_data,atom_type_pairs=[(0,1)])
# structural.plot_percent_z_displacements(zbox_data,zbox_data[f'{zbox_path}/{zbox_ref}'],atom_type_pairs=[(0,1)])

