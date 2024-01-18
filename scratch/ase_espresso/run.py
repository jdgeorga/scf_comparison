from ase.calculators.espresso import Espresso
from ase.io import read, write
import os

path_to_pseudopotentials = "/global/homes/j/jdgeorga/espresso/pseudo_new/"
binary_path = "/global/common/software/nersc/pm-2021q4/sw/qe/qe-7.0/bin/pw.x"

os.environ["ASE_ESPRESSO_COMMAND"] = "pw.x -in espresso.pwi > espresso.pwo"

atom = read("MoS2-Bilayer.xyz")
atom.cell[2, 2] = 20.0

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
                'ecutwfc': 25,
                'occupations': 'smearing',
                'smearing': 'gauss',
                'degauss': 0.005,
                #'input_dft': 'vdw-df-c09',
                'edir': 3,
                'emaxpos': 0.5
            },
            'electrons': {
                'conv_thr': 1.0e-8,
                'electron_maxstep': 1500,
                #'mixing_mode': 'local-TF',
                # 'mixing_beta': 0.3,
                # 'diagonalization': 'david',
            }}

# Pseudopotentials from SG15 ONCV library
pseudopotentials = {'Mo': 'Mo_ONCV_PBE-1.2.upf',
                    'S': 'S_ONCV_PBE-1.2.upf'}

calc = Espresso(
                input_data=input_data,
                command = f'pw.x -npool 1 -in espresso.pwi > espresso.pwo',
                pseudopotentials=pseudopotentials,
                pseudo_dir=path_to_pseudopotentials,
                tstress=False, tprnfor=True, kpts=(3, 3, 1),
                directory='test',
                label = 'espresso')
atom.calc = calc
print(atom.get_potential_energy())
print(atom.calc.get_forces())
