from ase.calculators.espresso import Espresso
from ase.io import read, write
from moirecompare.utils import (ase_to_phonopy_atoms,
                                phonopy_atoms_to_ase,
                                replace_line_starting_with)
from phonopy import Phonopy
import numpy as np
import os
from moirecompare.phonons import freeze_phonons


path_to_pseudopotentials = "/home1/08526/jdgeorga/espresso/pseudo_new/"
binary_path = "/global/common/software/nersc/pm-2021q4/sw/qe/qe-7.0/bin/pw.x"
# binary_path = "/global/common/software/nersc/pm-2021q4/sw/qe/pm-cpu/qe-7.0/bin"
binary_path="/global/homes/j/jdgeorga/codes/q-e/bin/pw.x"


os.environ["ASE_ESPRESSO_COMMAND"] = "/global/homes/j/jdgeorga/codes/q-e/bin/pw.x -in espresso.pwi > espresso.pwo"


def run_scf_base(ase_atom, input_data, pseudopotentials):
    print("Running SCF base")
    base_calc = Espresso(
        input_data=input_data,
        # command='ibrun -n 56 /global/homes/j/jdgeorga/codes/q-e/bin/pw.x -in scf.pwi > scf.pwo',
        pseudopotentials=pseudopotentials,
        pseudo_dir=path_to_pseudopotentials,
        tstress=False, tprnfor=True, kpts=(6, 6, 1),
        directory='disps/scf_base',
        label='scf')
    base_calc.write_input(ase_atom)
    replace_line_starting_with("disps/scf_base/scf.pwi",'   conv_thr         = 1e-10', "   conv_thr         = 1d-10" )
    # base_calc.read_results()
    # ase_atom.calc = base_calc

    # ase_atom.get_potential_energy()
    print("Done SCF base")
    return base_calc


def get_displacement_atoms(ase_atom):
    print("Getting displacement atoms")
    phonopy_atoms = ase_to_phonopy_atoms(ase_atom)

    supercell_n = 3
    # Initialize Phonopy object and generate displacements
    phonon = Phonopy(phonopy_atoms,
                     supercell_matrix=np.diag([supercell_n, supercell_n, 1]),
                     log_level=2)
    # phonon.generate_displacements(is_diagonal = False)
    phonon.generate_displacements(is_diagonal=True)
    phonon_displacement_list = phonon.get_supercells_with_displacements()
    phonon.save("phonon_no_forces.yaml")

    # Print the generated displacements
    for i, d in enumerate(phonon.displacements):
        print(f"Displacement {i}", d)

    # Convert PhonopyAtoms in the list to ASE Atoms objects
    ase_disp_list = [phonopy_atoms_to_ase(x) for x in phonon_displacement_list]

    # Write the displaced structures to new QE input files
    n_structures = len(ase_disp_list)

    return phonon, ase_disp_list


def run_scf_displaced(base_calc, ase_disp_list, input_dat, out_dir='disps'):
    print("Running SCF displaced")
    #input_data['electrons'].update({'startingpot': 'file'})

    for i, ase_disp in enumerate(ase_disp_list):

        print(f"Running SCF displaced {i}")

        input_data['control'].update({'prefix': 'scf_disp'})

        base_calc.directory = f"{out_dir}/scf_disp_{i}"
        #base_calc.input_data = input_data
        #print(base_calc.directory)

        #ref_dir = "disps/scf_base/scf_base.save"
        
        #os.system(f"mkdir -p {base_calc.directory}/scf_disp.save")
        #os.system(f"cp -p {ref_dir}/charge-density.dat {base_calc.directory}/scf_disp.save/.")
        #os.system(f"cp -p {ref_dir}/data-file-schema.xml {base_calc.directory}/scf_disp.save/.")

        base_calc.write_input(ase_disp)
        replace_line_starting_with(f"{out_dir}/scf_disp_{i}/scf.pwi",'   conv_thr         = 1e-10', "   conv_thr         = 1d-10" )

        print(f"Done SCF displaced {i}")
    return 0

def run_scf_layers(base_calc, ase_disp_list, input_dat, out_dir='disps'):
    print("Running SCF displaced")
    #input_data['electrons'].update({'startingpot': 'file'})

    for i, ase_disp in enumerate(ase_disp_list):

        print(f"Running SCF displaced {i}")

        input_data['control'].update({'prefix': 'scf_disp'})

        atomic_pos = ase_disp.positions.copy()
        print(ase_disp.cell[2,2])
        atomic_pos[atomic_pos[:,2] > ase_disp.cell[2,2]/2,2] -= ase_disp.cell[2,2]
        ase_disp.positions = atomic_pos.copy()
        

        # mid_point = ase_disp.positions[ase_disp.arrays['numbers'] == 42,2].mean()
        mid_point = ase_disp.positions[:,2].mean()
        print(mid_point)
        # Full
        base_calc.directory = f"{out_dir}/scf_disp_{i}/full"
        base_calc.write_input(ase_disp)
        replace_line_starting_with(f"{out_dir}/scf_disp_{i}/full/scf.pwi",'   conv_thr         = 1e-10', "   conv_thr         = 1d-10" )

        # Layer A
        L1_atom = ase_disp[ase_disp.positions[:,2]< mid_point]
        base_calc.directory = f"{out_dir}/scf_disp_{i}/L1"
        base_calc.write_input(L1_atom)
        replace_line_starting_with(f"{out_dir}/scf_disp_{i}/L1/scf.pwi",'   conv_thr         = 1e-10', "   conv_thr         = 1d-10" )
        # Layber B
        L2_atom = ase_disp[ase_disp.positions[:,2] >= mid_point]
        base_calc.directory = f"{out_dir}/scf_disp_{i}/L2"
        base_calc.write_input(L2_atom)
        replace_line_starting_with(f"{out_dir}/scf_disp_{i}/L2/scf.pwi",'   conv_thr         = 1e-10', "   conv_thr         = 1d-10" )

        print(f"Done SCF displaced {i}")
    return 0



def calculate_phonons(phonon, ase_disp_list, plot_bandstructure=False, out_dir='disps'):

    n_structures = len(phonon.displacements)

    atom_list = []

    # Read the atomic structures and forces from output files
    for num in np.arange(0, n_structures, dtype=int):
        atom_list.append(read(f"{out_dir}/scf_disp_{num}/scf.pwo"))

    # Set the forces for each displaced supercell and produce force constants
    phonon.forces = np.array([a.get_forces() for a in atom_list])
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.save("phonon_with_forces.yaml")

    mesh = [11, 11, 1]
    phonon.run_mesh(mesh, with_eigenvectors=True)
    phonon.write_hdf5_mesh()

    if plot_bandstructure:

        from moirecompare.utils import qpoints_Band_paths
        import sns
        plt = sns.plt
        sns.update_math_fonts()
        colors = sns.color_palette()

        print("Plotting phonon band structure")
        band_labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
        Band_points = 500

        Qpoints = np.array([[0.0000, 0.0000, 0.0000],
                            [0.50000, 0.00000, 0.00000],
                            [0.3333333, 0.33333333, 0.00000],
                            [0.0000, 0.0000, 0.0000]])

        # Generate the band points using high-symmetry qpoints
        bands = qpoints_Band_paths(Qpoints, Band_points)

        # Plot setup
        plt.figure(figsize=(15, 15))
        phonon.set_band_structure(bands, is_eigenvectors=True, labels=band_labels)

        phonon.write_hdf5_band_structure(filename=f"{out_dir}/bands.hdf5")
        # Scatter plot the phonon frequencies for each band
        for i in range(phonon.unitcell.get_number_of_atoms() * 3):
            plt.scatter(phonon.get_band_structure_dict()['distances'][0],
                        phonon.get_band_structure_dict()['frequencies'][0][:, i],
                        s=3, c='r')
            plt.scatter(phonon.get_band_structure_dict()['distances'][1],
                        phonon.get_band_structure_dict()['frequencies'][1][:, i],
                        s=3, c='r')
            plt.scatter(phonon.get_band_structure_dict()['distances'][2],
                        phonon.get_band_structure_dict()['frequencies'][2][:, i],
                        s=3, c='r')

        # Save the figure as a png file
        plt.savefig(f"phonon_bandstructure_{out_dir}.png")

        return 0


def main(ase_atom, input_data, pseudopotentials):

    base_calc = run_scf_base(ase_atom, input_data, pseudopotentials)

    # phonon, ase_disp_list = get_displacement_atoms(ase_atom)

    # run_scf_displaced(base_calc, ase_disp_list, input_data, out_dir='disps')

    # calculate_phonons(phonon, ase_disp_list, plot_bandstructure=True)

    frozen_atoms_list = freeze_phonons(n_sample=5, supercell_n=3)

    run_scf_layers(base_calc, frozen_atoms_list, input_data, out_dir='FF')


# main script

if __name__ == '__main__':

    ase_atom = read("MoS2-Bilayer.xyz")
    ase_atom.positions[:, 2] -= ase_atom.positions[0, 2]
    ase_atom.cell[2, 2] = 20.0

    input_data = {
                'control': {
                    'prefix': 'scf_base',
                    'calculation': 'scf',
                    'disk_io': 'low',
                    'verbosity': 'high',
                    'tefield': True,
                    'dipfield': True
                },
                'system': {
                    'ecutwfc': 70,
                    'occupations': 'smearing',
                    'smearing': 'gauss',
                    'degauss': 0.01,
                    'input_dft': 'vdw-df-c09',
                    'edir': 3,
                    'emaxpos': 0.5
                },
                'electrons': {
                    'conv_thr': 1.0e-10,
                    'electron_maxstep': 1000,
                    'mixing_mode': 'local-TF',
                    'mixing_beta': 0.3,
                    'diagonalization': 'david',
                    'diago_david_ndim': 8,
                }}

    # Pseudopotentials from SG15 ONCV library
    pseudopotentials = {'Mo': 'Mo_ONCV_PBE-1.2.upf',
                        'S': 'S_ONCV_PBE-1.2.upf'}
    main(ase_atom, input_data, pseudopotentials)

# calc = Espresso(
#                 input_data=input_data,
#                 command = f'pw.x -npool 1 -in espresso.pwi > espresso.pwo',
#                 pseudopotentials=pseudopotentials,
#                 pseudo_dir=path_to_pseudopotentials,
#                 tstress=False, tprnfor=True, kpts=(3, 3, 1),
#                 directory='test',
#                 label = 'espresso')
# atom.calc = calc
# print(atom.get_potential_energy())
# print(atom.calc.get_forces())