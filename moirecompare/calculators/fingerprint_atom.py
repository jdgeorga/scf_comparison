from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
import numpy as np
from scipy.spatial import cKDTree

def create_moire_grid(moire_atoms, n, padding_factor=0.0):
    """
    Subdivides a moiré structure into an n x n grid of atom objects.

    Parameters:
    - moire_atoms: The atoms object representing the moiré structure.
    - n: The number of subdivisions along one axis.
    - padding_factor: The amount of padding to add to the grid cell.
                      1.0 adds an additional grid cell to each side 
                      of the grid cell.

    Returns:
    - grid: A 2D array representing the grid points.
    - moire_grid_atoms: A 2D array of atoms objects for each grid cell.
    """
    # Store the original moiré cell
    original_cell = moire_atoms.cell

    # Repeat the atoms to ensure full coverage for edge cases
    moire_atoms = moire_atoms.repeat((3, 3, 1))

    # Restore the original cell size to keep physical dimensions consistent
    moire_atoms.set_cell(original_cell)

    # Calculate the shift needed for moiré pattern
    moire_shift = moire_atoms.cell[0] + moire_atoms.cell[1]
    moire_atoms.positions[:, :2] -= moire_shift[:2]

    # Create grid points within the moiré cell
    x = (np.linspace(0, 1, n + 1) - 1 / (2 * n))[:, None] * moire_atoms.cell[0]
    y = (np.linspace(0, 1, n + 1) - 1 / (2 * n))[:, None] * moire_atoms.cell[1]

    # Define the size of each grid cell
    grid_cell = moire_atoms.cell / np.array([n, n, 1])

    # Combine x and y to form a 2D grid
    grid = x[:, None, :2] + y[None, :, :2]

    # Transform the grid points to the moiré coordinates
    moire_grid = grid @ np.linalg.inv(grid_cell[:2, :2])

    # Initialize a list to hold the atoms objects for each grid cell
    moire_grid_atoms = []

    # Count the unique types of atoms in the moiré structure
    n_unique_atom_types = len(np.unique(moire_atoms.arrays['atom_types']))

    # Iterate over the grid to populate it with atoms objects
    for i in range(n):
        grid_atoms_row = []
        for j in range(n):
            # Copy the moiré atoms to manipulate them for each grid cell
            m_ij = moire_atoms.copy()

            # Define the corners of the current grid cell in moiré coordinates
            x0y0 = moire_grid[i, j].copy()
            x1y1 = moire_grid[i + 1, j + 1].copy()

            # Apply padding to the grid cell
            dxy = np.array(x1y1 - x0y0)
            x0y0 -= dxy * padding_factor
            x1y1 += dxy * padding_factor

            # Get the positions of atoms in the current moiré atoms object
            m_ij_pos = m_ij.get_positions()
            m_ij_pos_scaled = m_ij_pos[:, :2] @ np.linalg.inv(grid_cell[:2, :2])

            # Filter atoms to include only those within the current grid cell
            m_ij = m_ij[np.logical_and(m_ij_pos_scaled[:, 0] < x1y1[0],
                                       m_ij_pos_scaled[:, 0] > x0y0[0])]

            # Update positions after filtering
            m_ij_pos_scaled = m_ij_pos_scaled[np.logical_and(m_ij_pos_scaled[:, 0] < x1y1[0],
                                                             m_ij_pos_scaled[:, 0] > x0y0[0])]

            # Repeat filtering for the y dimension
            m_ij = m_ij[np.logical_and(m_ij_pos_scaled[:, 1] < x1y1[1],
                                       m_ij_pos_scaled[:, 1] > x0y0[1])]

            # Check if the filtered atoms contain all unique atom types
            if len(np.unique(m_ij.arrays['atom_types'])) != n_unique_atom_types:
                print(f"WARNING: Grid cell ({i}, {j}) contains {len(np.unique(m_ij.arrays['atom_types']))} unique atom types")
                return None

            # Append the filtered atoms object to the current row
            grid_atoms_row.append(m_ij)

        # Append the current row to the moire_grid_atoms
        moire_grid_atoms.append(grid_atoms_row)

    return grid, moire_grid_atoms

def create_atom_centered_moire_grid(moire_atoms, n, padding_factor=0.0):
    """
    Subdivides a moiré structure into an n x n grid of atom objects.

    Parameters:
    - moire_atoms: The atoms object representing the moiré structure.
    - n: The number of subdivisions along one axis.
    - padding_factor: The amount of padding to add to the grid cell.
                      1.0 adds an additional grid cell to each side 
                      of the grid cell.

    Returns:
    - grid: A 2D array representing the grid points.
    - moire_grid_atoms: A 2D array of atoms objects for each grid cell.
    """
    # Store the original moiré cell
    original_cell = moire_atoms.cell

    # Repeat the atoms to ensure full coverage for edge cases
    moire_atoms = moire_atoms.repeat((3, 3, 1))

    # Restore the original cell size to keep physical dimensions consistent
    moire_atoms.set_cell(original_cell)

    # Calculate the shift needed for moiré pattern
    moire_shift = moire_atoms.cell[0] + moire_atoms.cell[1]
    moire_atoms.positions[:, :2] -= moire_shift[:2]

    # Create grid points within the moiré cell
    x = (np.linspace(0, 1, n + 1) - 1 / (2 * n))[:, None] * moire_atoms.cell[0]
    y = (np.linspace(0, 1, n + 1) - 1 / (2 * n))[:, None] * moire_atoms.cell[1]

    # Define the size of each grid cell
    grid_cell = moire_atoms.cell / np.array([n, n, 1])

    # Combine x and y to form a 2D grid
    grid = x[:, None, :2] + y[None, :, :2]

    # Transform the grid points to the moiré coordinates
    moire_grid = grid @ np.linalg.inv(grid_cell[:2, :2])

    # Initialize a list to hold the atoms objects for each grid cell
    moire_grid_atoms = []

    # Count the unique types of atoms in the moiré structure
    n_unique_atom_types = len(np.unique(moire_atoms.arrays['atom_types']))

    # Iterate over the grid to populate it with atoms objects
    for i in range(n):
        grid_atoms_row = []
        for j in range(n):
            # Copy the moiré atoms to manipulate them for each grid cell
            m_ij = moire_atoms.copy()

            # Define the corners of the current grid cell in moiré coordinates
            x0y0 = moire_grid[i, j].copy()
            x1y1 = moire_grid[i + 1, j + 1].copy()

            # Apply padding to the grid cell
            dxy = np.array(x1y1 - x0y0)
            x0y0 -= dxy * padding_factor
            x1y1 += dxy * padding_factor

            # Get the positions of atoms in the current moiré atoms object
            m_ij_pos = m_ij.get_positions()
            m_ij_pos_scaled = m_ij_pos[:, :2] @ np.linalg.inv(grid_cell[:2, :2])

            # Filter atoms to include only those within the current grid cell
            m_ij = m_ij[np.logical_and(m_ij_pos_scaled[:, 0] < x1y1[0],
                                       m_ij_pos_scaled[:, 0] > x0y0[0])]

            # Update positions after filtering
            m_ij_pos_scaled = m_ij_pos_scaled[np.logical_and(m_ij_pos_scaled[:, 0] < x1y1[0],
                                                             m_ij_pos_scaled[:, 0] > x0y0[0])]

            # Repeat filtering for the y dimension
            m_ij = m_ij[np.logical_and(m_ij_pos_scaled[:, 1] < x1y1[1],
                                       m_ij_pos_scaled[:, 1] > x0y0[1])]

            # Check if the filtered atoms contain all unique atom types
            if len(np.unique(m_ij.arrays['atom_types'])) != n_unique_atom_types:
                print(f"WARNING: Grid cell ({i}, {j}) contains {len(np.unique(m_ij.arrays['atom_types']))} unique atom types")
                return None

            # Append the filtered atoms object to the current row
            grid_atoms_row.append(m_ij)

        # Append the current row to the moire_grid_atoms
        moire_grid_atoms.append(grid_atoms_row)

    return grid, moire_grid_atoms


class BaseFingerprintCalculator(Calculator):
    implemented_properties = ['energy', 'energies', 'forces']  # Define the properties this calculator can compute

    def __init__(self, dataset, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.dataset = dataset
        self.dataset_atom_types = np.unique(self.dataset[0].arrays['atom_types'])
        self.db_fingerprints = self.get_db_local_2Dfingerprint(self.dataset)

    def calculate(self,
                  local_structure: Atoms,
                  properties=None,
                  system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, local_structure, properties, system_changes)
        local_fingerprint = self.get_local_2Dfingerprint(local_structure, k=12)
        best_match_idx = self.match_fingerprints(local_fingerprint)

        best_match_energy = self.dataset[best_match_idx].info['energy']
        best_match_forces = self.dataset[best_match_idx].arrays['forces']
        bandgap_rel = self.dataset[best_match_idx].info['bandgap_rel']
        bandgap = self.dataset[best_match_idx].info['bandgap']

        self.results = {}
        self.results['energy'] = best_match_energy
        self.results['forces'] = best_match_forces
        self.results['best_match_idx'] = best_match_idx
        self.results['fingerprint'] = local_fingerprint
        self.results['bandgap_rel'] = bandgap_rel
        self.results['bandgap'] = bandgap

    def get_local_2Dfingerprint(self, local_structure, k: int = 12):
        n_neigbors = k

        zero_atom = local_structure[local_structure.arrays['atom_types'] == 0]
        zero_atom_position = zero_atom[np.argmin(np.linalg.norm(zero_atom.positions[:, :2]))].position
        local_structure.positions[:, :2] -= zero_atom_position[:2]

        local_structure_padded = local_structure.repeat([1, 1, 1])
        local_structure_shift = local_structure.cell[0] + local_structure.cell[1]
        # local_structure_padded.positions -= local_structure_shift

        unique_atom_types = np.unique(local_structure.arrays['atom_types'])

        tree = cKDTree(local_structure_padded.positions[:, :2])

        including_all_atom_type_cond = False
        while including_all_atom_type_cond == False:
            if n_neigbors > len(local_structure):
                n_neigbors = len(local_structure)

            _, ind = tree.query(local_structure.positions[:, :2],
                                k=n_neigbors)

            unique_atoms_at_ind = [len(np.unique(i)) for i in local_structure_padded.arrays['atom_types'][ind]]
            including_all_atom_type_cond = np.array(unique_atoms_at_ind) == len(unique_atom_types)
            including_all_atom_type_cond = including_all_atom_type_cond.all()
            n_neigbors *= 2

        xy_diff = local_structure_padded.positions[ind, :2] - local_structure.positions[:, None, :2]
        z_diff = local_structure_padded.positions[ind, 2] - local_structure.positions[:, None, 2]

        xy_diff_norm = np.linalg.norm(xy_diff, axis=-1)
        z_diff_norm = np.abs(z_diff)

        ntype_dists = np.concatenate((local_structure_padded.arrays['atom_types'][ind][:,:,None],
                                      xy_diff_norm[:, :, None]),
                                      axis=-1)
        ntype_dists = np.take_along_axis(ntype_dists[:,:,:],
                                            np.argsort(ntype_dists[:,:,0],axis = 1)[:,:,None],
                                            axis=1)

        ntype_z_dists = np.concatenate((local_structure_padded.arrays['atom_types'][ind][:,:,None],
                                        z_diff_norm[:,:,None]),
                                        axis = -1)
        ntype_z_dists = np.take_along_axis(ntype_z_dists[:,:,:],
                                            np.argsort(ntype_z_dists[:,:,0],axis = 1)[:,:,None],
                                            axis=1)

        reduced_ntype_dists = [np.split(type[:, 1], np.unique(type[:, 0], return_index=True)[1][1:]) for type in ntype_dists] 
        reduced_ntype_z_dists = [np.split(type[:, 1], np.unique(type[:, 0], return_index=True)[1][1:]) for type in ntype_z_dists] 

        reduced_ntype_fingerprint = [[[np.min(m),np.min(l)]for m,l in zip(type,type_z)] for type, type_z in zip(reduced_ntype_dists,reduced_ntype_z_dists)]
        reduced_ntype_fingerprint = np.concatenate((local_structure.arrays['atom_types'][:,None,None].repeat(len(unique_atom_types),axis = 1),
                                                reduced_ntype_fingerprint),axis = -1).reshape(-1,len(unique_atom_types),3)
        reduced_ntype_fingerprint = np.take_along_axis(reduced_ntype_fingerprint[:,:,:],
                                                    np.argsort(reduced_ntype_fingerprint[:,:,0],axis = 0)[:,:,None],
                                                    axis=0)

        reduced_nxn_dist_fingerprint = np.split(reduced_ntype_fingerprint[:, :, 1:], np.unique(reduced_ntype_fingerprint[:,0,0], return_index=True)[1][1:])
        reduced_nxn_dist_fingerprint = np.array([np.min(type, axis = 0) for type in reduced_nxn_dist_fingerprint])
   
        return np.array(reduced_nxn_dist_fingerprint)
    
    def get_db_local_2Dfingerprint(self, db, k: int = 12):
        reduced_nxn_dist_fingerprints = []
        for s in db:
            n_neigbors = k
            s_padded = s.repeat([7, 7, 1])
            
            s_shift = s.cell[0] + s.cell[1]
            s_padded.positions -= 3*s_shift

            unique_atom_types = np.unique(s.arrays['atom_types'])
            
            tree = cKDTree(s_padded.positions[:, :2])

            including_all_atom_type_cond = False
            while including_all_atom_type_cond == False:
                _, ind = tree.query(s.positions[:,:2], k = n_neigbors)

                unique_atoms_at_ind = [len(np.unique(i)) for i in s_padded.arrays['atom_types'][ind]]
                including_all_atom_type_cond = np.array(unique_atoms_at_ind) == len(unique_atom_types)
                including_all_atom_type_cond = including_all_atom_type_cond.all()
                n_neigbors *= 2

            xy_diff = s_padded.positions[ind,:2] - s.positions[:,None,:2]
            z_diff = s_padded.positions[ind,2] - s.positions[:,None,2]

            xy_diff_norm = np.linalg.norm(xy_diff,axis = -1)
            z_diff_norm = np.abs(z_diff)

            ntype_dists = np.concatenate((s_padded.arrays['atom_types'][ind][:,:,None],xy_diff_norm[:,:,None]),axis = -1)
            ntype_dists = np.take_along_axis(ntype_dists[:,:,:], np.argsort(ntype_dists[:,:,0],axis = 1)[:,:,None], axis=1)

            ntype_z_dists = np.concatenate((s_padded.arrays['atom_types'][ind][:,:,None],z_diff_norm[:,:,None]),axis = -1)
            ntype_z_dists = np.take_along_axis(ntype_z_dists[:,:,:], np.argsort(ntype_z_dists[:,:,0],axis = 1)[:,:,None], axis=1)

            # ntype_dists = ntype_dists[ntype_dists[:, 0].argsort()]
            reduced_ntype_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_dists] 
            reduced_ntype_z_dists = [np.split(metal[:, 1], np.unique(metal[:, 0], return_index=True)[1][1:]) for metal in ntype_z_dists]
            reduced_ntype_fingerprint = [[[np.min(m),np.min(l)]for m,l in zip(metal,metal_z)] for metal, metal_z in zip(reduced_ntype_dists,reduced_ntype_z_dists)]

            # print(s.arrays['atom_types'][:,None,None].shape)
            reduced_ntype_fingerprint = np.concatenate((s.arrays['atom_types'][:,None,None].repeat(len(unique_atom_types),axis = 1),
                                                reduced_ntype_fingerprint),axis = -1).reshape(-1,len(unique_atom_types),3)
            reduced_ntype_fingerprint = np.take_along_axis(reduced_ntype_fingerprint[:,:,:],
                                                        np.argsort(reduced_ntype_fingerprint[:,:,0],axis = 0)[:,:,None],
                                                        axis=0)

            reduced_nxn_dist_fingerprint = np.split(reduced_ntype_fingerprint[:, :, 1:], np.unique(reduced_ntype_fingerprint[:,0,0], return_index=True)[1][1:])
            reduced_nxn_dist_fingerprint = np.array([np.min(type, axis = 0) for type in reduced_nxn_dist_fingerprint])

            reduced_nxn_dist_fingerprints.append(reduced_nxn_dist_fingerprint)
        print(np.array(reduced_nxn_dist_fingerprint).shape)
        return np.array(reduced_nxn_dist_fingerprints)


    def match_fingerprints(self,
                           moire_point_fingerprint,
                           plot: bool = False):

        ref = self.db_fingerprints.reshape(-1, self.db_fingerprints[0].size)
        print(ref.shape)
        tree = cKDTree(ref)
        print(moire_point_fingerprint.shape)
        _, idx = tree.query(moire_point_fingerprint.reshape(-1,
                                                            self.db_fingerprints[0].size))
        return idx[0]

    
if __name__ == "__main__":
    from ase.io import read
    from fingerprint import (
        prepare_database,
        create_grid,
        BaseFingerprintCalculator
    )
    from moirecompare.utils import add_allegro_number_array
    import matplotlib.pyplot as plt

    dataset = read("./mos2_interlayer_dset.xyz", index=":")
    # moire_structure = read("relax_546.out", index=-1, format="espresso-out")
    # moire_structure = read("scf_1014.out", index=-1, format="espresso-out")
    moire_structure = read("MoS2-2deg-relax.xsf", format="xsf")

    moire_structure.arrays['atom_types'] = add_allegro_number_array(moire_structure)

    dataset = prepare_database(dataset)

    n = 24
    moire_grid, moire_grid_atoms = create_grid(moire_structure, n)
    moire_grid_cell = moire_structure.cell / np.array([n, n, 1])

    calc = BaseFingerprintCalculator(dataset)

    moire_grid_gap_vals = []
    for i in range(len(moire_grid_atoms)):
        moire_grid_gap_vals_row = []
        for j in range(len(moire_grid_atoms[i])):
            moire_grid_atoms[i][j].cell = moire_grid_cell
            moire_grid_atoms[i][j].calc = calc
            moire_grid_atoms[i][j].calc.calculate(moire_grid_atoms[i][j])

            moire_grid_gap_vals_row.append(moire_grid_atoms[i][j].calc.results['bandgap_rel'])

        moire_grid_gap_vals.append(moire_grid_gap_vals_row)

    n_rows = len(moire_grid_atoms)

    plt.figure(figsize=(10, 10))

    vmin = np.array(moire_grid_gap_vals).min()
    vmax = np.array(moire_grid_gap_vals).max()    

    offsets = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
    offsets = offsets + 1/(2*n)
    offsets = offsets @ moire_structure.cell[:2, :2]

    for i in range(n_rows):
        for j in range(n_rows):
            x0y0 = moire_grid[i][j]
            x0y0 = x0y0 + (offsets * 1.)
            plot1 = plt.scatter(x0y0[:, 0],
                                x0y0[:, 1],
                                c=[moire_grid_gap_vals[i][j]]*9,
                                s=180, marker="h",
                                vmin=vmin, vmax=vmax,
                                cmap='plasma')

    p1 = np.array([0, 0])
    p2 = moire_structure.cell[0]
    p3 = moire_structure.cell[0] + moire_structure.cell[1]
    p4 = moire_structure.cell[1]

    # Draw lines around each grid cell
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black')  # Bottom edge
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black')  # Right edge
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color='black')  # Top edge
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], color='black')  # Left edge
    # plt.xlim(moire_structure.cell[1, 0]-1,
    #          moire_structure.cell[0, 0]+1)
    # plt.ylim(moire_structure.cell[0, 1]-5,
    #          moire_structure.cell[1, 1]+5)
    plt.colorbar(plot1, fraction=0.03,
                 label='Relative Bandgap (eV)')
    plt.title("Relaxed MoS2/MoS2 bilayer: Relative Bandgap")
    plt.axis('scaled')
    plt.xlim(p4[0] - 5, p2[0] + 5)
    plt.ylim(p1[0] - 5, p3[1] + 5)
    plt.savefig("moire_grid_bandgap.png", dpi=300, bbox_inches='tight')









        