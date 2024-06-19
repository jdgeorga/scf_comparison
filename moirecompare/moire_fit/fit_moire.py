from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from moirecompare.utils import add_allegro_number_array
from ase import Atoms
from scipy.spatial import cKDTree
import pymoire as pm
import argparse


def prepare_database(structures, pristine_atoms):

    pristine_cell = pristine_atoms.cell
    pristine_cell[2, 2] = 100

    clean_database = []
    # unique_atom_types = np.unique(structures[0].arrays['atom_types'])
    for ss in structures:
        s = ss.copy()
        del s[[3,4,5,9,10,11]]
        s.positions -= s.positions[np.where(s.arrays['atom_types'] == 0)[0][0]]
        s.arrays['atom_types'] = np.array([0, 1, 2, 3, 4, 5],dtype=int)
        s.set_cell(pristine_cell)
        clean_database.append(s)
        # scaled_positions = s.get_scaled_positions()

    return clean_database


def zero_shell_fixed(x, n_shell: int = 1, eps=1e-6):
    '''
    Fits crystal coordinates between (-0.5 - n_shell + 1) and (.5 + n_shell + 1)
        
    Parameters
    ----------
    x: float
        Crystal coordinate
    n_shell: int
        Number of shells
    eps: float
        Error term
    '''
    y = x.copy()
    y[np.logical_and(y>( 0.5  - n_shell ),
                     y<( 0.5 - n_shell + eps))] = 0.5 - n_shell
    y = (y + n_shell - 0.5 ) % (2*n_shell -1 )  - n_shell + 0.5
    y[y>(-0.5 + n_shell  - eps)] = n_shell - 0.5
    return y


def reduce_to_shell(positions,pristine_atom_cell, n_shell = 1):
    crystal_diff = positions @ np.linalg.inv(pristine_atom_cell)
    zero_shell_crystal = zero_shell_fixed(crystal_diff, n_shell = n_shell) 
    zero_shell_cart = zero_shell_crystal @ pristine_atom_cell
    return zero_shell_cart


def get_moire_point_2Dfingerprint(moire, pristine_atoms_cell, atom_type: int = 0, k: int = 12):

    moire_padded = moire.repeat([3, 3, 1])
    moire_shift = moire.cell[0] + moire.cell[1]
    moire_padded.positions -= moire_shift
 
    tree = cKDTree(moire_padded.positions[:,:2])

    dist, ind = tree.query(moire[moire.arrays['atom_types'] == atom_type].positions[:,:2], k = k)

    pos_diff = moire_padded.positions[ind,:2] - moire[moire.arrays['atom_types'] == atom_type].positions[:,None,:2]
    z_diff = moire_padded.positions[ind,2] - moire[moire.arrays['atom_types'] == atom_type].positions[:,None,2]

    zero_shell_cart = reduce_to_shell(pos_diff,pristine_atoms_cell[:2,:2], n_shell = 1)

    zero_shell_norm = np.linalg.norm(zero_shell_cart,axis = -1)

    ntype_dists = np.concatenate((moire_padded.arrays['atom_types'][ind][:,:,None],zero_shell_norm[:,:,None]),axis = -1)
    ntype_dists = np.take_along_axis(ntype_dists[:,:,:], np.argsort(ntype_dists[:,:,0],axis = 1)[:,:,None], axis=1)

    ntype_z_dists = np.concatenate((moire_padded.arrays['atom_types'][ind][:,:,None],z_diff[:,:,None]),axis = -1)
    ntype_z_dists = np.take_along_axis(ntype_z_dists[:,:,:], np.argsort(ntype_z_dists[:,:,0],axis = 1)[:,:,None], axis=1)

    reduced_ntype_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_dists] 
    reduced_ntype_z_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_z_dists] 

    reduced_ntype_figureprint = [[[np.mean(m),np.mean(l)]for m,l in zip(metal, metal_z)] for metal, metal_z in zip(reduced_ntype_dists,reduced_ntype_z_dists)]

    return reduced_ntype_figureprint

def get_db_point_2Dfingerprint(db, pristine_atoms_cell, atom_type:int = 0, k:int = 12):
    reduced_ntype_figureprints = []
    for s in db:
        s_padded = s.repeat([5,5,1])
        s_shift = s.cell[0] + s.cell[1]
        s_padded.positions -= 2*s_shift
        
        tree = cKDTree(s_padded.positions[:,:2])

        _, ind = tree.query(s[s.arrays['atom_types'] == atom_type].positions[:,:2], k = k)

        pos_diff = s_padded.positions[ind,:2] - s[s.arrays['atom_types'] == atom_type].positions[:,None,:2]
        z_diff = s_padded.positions[ind,2] - s[s.arrays['atom_types'] == atom_type].positions[:,None,2]

        zero_shell_cart = reduce_to_shell(pos_diff,pristine_atoms_cell[:2,:2], n_shell = 1)

        zero_shell_norm = np.linalg.norm(zero_shell_cart,axis = -1)

        ntype_dists = np.concatenate((s_padded.arrays['atom_types'][ind][:,:,None],zero_shell_norm[:,:,None]),axis = -1)
        ntype_dists = np.take_along_axis(ntype_dists[:,:,:], np.argsort(ntype_dists[:,:,0],axis = 1)[:,:,None], axis=1)

        ntype_z_dists = np.concatenate((s_padded.arrays['atom_types'][ind][:,:,None],z_diff[:,:,None]),axis = -1)
        ntype_z_dists = np.take_along_axis(ntype_z_dists[:,:,:], np.argsort(ntype_z_dists[:,:,0],axis = 1)[:,:,None], axis=1)

        # ntype_dists = ntype_dists[ntype_dists[:, 0].argsort()]
        reduced_ntype_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_dists] 
        reduced_ntype_z_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_z_dists]
        reduced_ntype_figureprint = [[[np.mean(m), np.mean(l)]for m, l in zip(metal, metal_z)] for metal, metal_z in zip(reduced_ntype_dists, reduced_ntype_z_dists)]
        reduced_ntype_figureprints.append(reduced_ntype_figureprint[0])
    return reduced_ntype_figureprints


def match_fingerprints(clean_database,
                       moire,
                       db_point_fingerprint,
                       moire_point_fingerprint,
                       atom_type: int = 0,
                       plot: bool = False):

    ref = np.array(db_point_fingerprint).reshape(-1, 12)

    tree = cKDTree(ref)
    _, ind = tree.query(np.array(moire_point_fingerprint).reshape(-1, 12))

    e_vals = [clean_database[idx].info['bandgap_rel'] for idx in ind]

    if plot:
        m = moire[moire.arrays['atom_types'] == atom_type].repeat([3,3,1])
        moire_shift = moire.cell[0] + moire.cell[1]
        m.positions -= moire_shift
        plot1 = plt.scatter(m.positions[:,0],m.positions[:,1],s = 500,c = e_vals * 9 , cmap = 'plasma', marker = 'H')
        plt.axis('scaled')

        p1 = np.array([0,0])
        p2 = moire.cell[0]
        p3 = moire.cell[0] + moire.cell[1]
        p4 = moire.cell[1]
        # print(x0,y0)
        # Draw lines around each grid cell
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black') # Bottom edge
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black') # Right edge
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color='black') # Top edge
        plt.plot([p4[0], p1[0]], [p4[1], p1[1]], color='black') # Left edge
        plt.xlim(moire.cell[1,0]-1, moire.cell[0,0]+1)
        plt.ylim(moire.cell[0,1]-5, moire.cell[1,1]+5)
        plt.colorbar(plot1, fraction=0.03, label='Relative Band Gap (eV)')
        plt.savefig('moire_bandgap.png', dpi=300, bbox_inches='tight')


def main(structures_path, moire_path, layer1, layer2, atom_type):
    structures = read(structures_path, index=":")
    moire = read(moire_path, index=-1, format="espresso-out")
    moire.arrays['atom_types'] = add_allegro_number_array(moire)

    p = pm.materials.get_materials_db_path()
    pristine_atoms = pm.read_monolayer(p / 'MoS2.cif')

    clean_database = prepare_database(structures, pristine_atoms)

    moire_point_fingerprint = get_moire_point_2Dfingerprint(moire,
                                                            pristine_atoms.cell, 
                                                            atom_type=atom_type, k=12)
    db_point_fingerprint = get_db_point_2Dfingerprint(clean_database[:], pristine_atoms.cell,
                                                      atom_type=atom_type, k=12)

    match_fingerprints(clean_database, moire, db_point_fingerprint,
                       moire_point_fingerprint,
                       atom_type=atom_type, plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process structures and moire patterns.")
    parser.add_argument("--structures", type=str, required=True, help="Path to the structures file.")
    parser.add_argument("--moire", type=str, required=True, help="Path to the moire pattern file.")
    parser.add_argument("--layer1", type=str, required=True, help="Information for layer 1.")
    parser.add_argument("--layer2", type=str, required=True, help="Information for layer 2.")
    parser.add_argument("--atom_type", type=int, required=True, help="Atom type to be considered.")

    args = parser.parse_args()

    main(args.structures, args.moire, args.layer1, args.layer2, args.atom_type)


