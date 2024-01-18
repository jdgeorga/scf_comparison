from phonopy import load
from ase.io import read, write
import numpy as np
from moirecompare.utils import phonopy_atoms_to_ase, gen_ase
import sys

import numpy as np
import h5py
import numpy as np

import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
from scipy.spatial import cKDTree
from phonopy.units import THzToEv

import sns
plt = sns.plt
sns.update_math_fonts()

##########################
c_light = 299792458
h_planck = 6.62606896e-34
Bohr_to_Å = 0.52917725
hbar = 6.582e-16
Ryd = 4.35974394e-18 / 2.0
time_ry = h_planck / (2.0 * np.pi) / Ryd
Hz_to_kayser = 1.0e-2 / (2.0 * np.pi * c_light)
Ry_to_kayser = Hz_to_kayser / time_ry
Ry_to_eV = 13.605693123
mp_in_me = 918.07638 
THz_to_eV = 4.136 * 1e-3
k = 8.617333262e-5

def in_kayser(x):
    return x * Ry_to_kayser

def prefactor_in_Å(N_p,w_in_Ry, mass_in_mp):
    pre = np.sqrt(1 / (2 * N_p * w_in_Ry[None,None,:,:] * mp_in_me * mass_in_mp[None,:,None,None])) # Bohr
    pre *= Bohr_to_Å # Å
    return pre
#######################


def τ_from_νq(i_ν,i_q):
    print(f"i_ν: {i_ν}, w_νq: {w_νq[i_ν,i_q]}")


    pre_F = prefactor_in_Å(N_p,w_νq*THzToEv/Ry_to_eV, M_κ)
    # τ = pre_F[:,:,i_ν,i_q] * 2 * np.real((np.exp(1.0j * R_p @ qs[i_q].T )[:,None,None,i_q] * modes[:,:,i_ν,i_q]))
    τ = pre_F[:,:,i_ν,i_q] * 2 * np.real((np.exp(1.0j * R_p @ qs[i_q,:].T)[:,None] * modes[:,:,i_ν,i_q]))
    return τ


def n_from_νq(i_ν,i_q,T):
    return 1/(np.exp(w_νq[i_ν,i_q] * THzToEv / (k*T)) - 1)


def freeze_phonons(n_sample = 5, 
                   supercell_n = 3, 
                   mesh_file = "mesh.hdf5",
                   phonon_file = "phonon_with_forces.yaml",
                   ):

    # Get phonon eigs

    supercell_n = 3 

    eigs = h5py.File("mesh.hdf5")

    phonon = load("phonon_with_forces.yaml")
    atomic_mass_array = phonon.unitcell.get_masses().repeat(3)
    atomic_number_array = phonon.supercell.get_atomic_numbers()

    X,Y = np.meshgrid([0,1,2],[0,1,2])
    R_p = np.array([X.flatten(),Y.flatten(),np.zeros_like(X.flatten())]).T
    R_p = R_p @ phonon.unitcell.cell

    N_p = supercell_n**2
    M_κ = atomic_mass_array

    unit_atoms = phonon.unitcell.get_number_of_atoms()
    w, m = eigs['frequency'][()], eigs['eigenvector'][()]

    modes = m.transpose((1,2,0))[None,:,:,:]
    w_νq = w.copy().T.reshape((unit_atoms * 3,-1),order = 'F')
    w_νq[w_νq<0.] = 1e-12

    reclat = 2*np.pi*np.linalg.inv(phonon.primitive.cell).T
    qs = np.dot(eigs['qpoint'][()],reclat)


    n_q = modes.shape[3]
    n_ν = modes.shape[2]

    rand_q = np.random.randint(low = 0,high = n_q,size= n_sample)
    rand_ν = np.random.randint(low = 0,high = n_ν,size= n_sample)
    while (rand_q == 0).any() and (rand_ν <= 3).any():
        rand_q = np.random.randint(low = 0,high = n_q,size= n_sample)
        rand_ν = np.random.randint(low = 0,high = n_ν,size= n_sample)

    rand_τ = τ_from_νq(rand_ν, rand_q).transpose((2, 0, 1)) # (n_sample,p,κ)
    print(f"rand_max = {rand_τ.max()}")

    cell_out = phonon.supercell.cell
    # cell_out[2,2] = 30.000000
    new_atom_list = []
    print(phonon.supercell.positions[0])
    for t in rand_τ:
        new_pos = phonon.supercell.positions + t.reshape(-1, 3)
        new_pos_scaled = new_pos @ np.linalg.inv(cell_out)
        atom = gen_ase(cell_out, new_pos_scaled, atomic_number_array)
        new_atom_list.append(atom)

    return new_atom_list

