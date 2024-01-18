from ase.calculators.espresso import Espresso
from ase.io import read, write
from moirecompare.utils import (ase_to_phonopy_atoms,
                                phonopy_atoms_to_ase)
from phonopy import Phonopy
import numpy as np
import os
from moirecompare.phonons import freeze_phonons
from moirecompare.phonons import run_scf_base, run_scf_displaced


path_to_pseudopotentials = "/global/homes/j/jdgeorga/espresso/pseudo_new/"
binary_path = "/global/common/software/nersc/pm-2021q4/sw/qe/qe-7.0/bin/pw.x"
binary_path = "/global/common/software/nersc/pm-2021q4/sw/qe/pm-cpu/qe-7.0/bin"


os.environ["ASE_ESPRESSO_COMMAND"] = "pw.x -in espresso.pwi > espresso.pwo"


def main(ase_atom, input_data, pseudopotentials):

    base_calc = run_scf_base(ase_atom, input_data, pseudopotentials)

    frozen_atoms_list = freeze_phonons(n_sample=5, supercell_n=3)

    run_scf_displaced(base_calc, frozen_atoms_list, input_data, out_dir='FF')


# main script

if __name__ == '__main__':

    ase_atom = read("MoS2-Bilayer.xyz")
    ase_atom.positions[:, 2] -= ase_atom.positions[0, 2]
    ase_atom.cell[2, 2] = 16.0

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
                    'degauss': 0.005,
                    'input_dft': 'vdw-df-c09',
                    'edir': 3,
                    'emaxpos': 0.5
                },
                'electrons': {
                    'conv_thr': 1.0e-8,
                    'electron_maxstep': 1500,
                    'mixing_mode': 'local-TF',
                    'mixing_beta': 0.3,
                    'diagonalization': 'david',
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
