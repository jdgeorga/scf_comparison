from ase.io import read
from moirecompare.calculators.fingerprint import (
    create_moire_grid,
    BaseFingerprintCalculator
)
from moirecompare.utils import add_allegro_number_array
import matplotlib.pyplot as plt
import numpy as np

def prepare_database(structures):
    clean_database = []
    for ss in structures:
        s = ss
        del s[[3,4,5,9,10,11]]
        s.positions -= s.positions[np.where(s.arrays['atom_types'] == 0)[0][0]]
        s.arrays['atom_types'] = np.array([0, 1, 2, 3, 4, 5],dtype=int)
        clean_database.append(s)

    return clean_database

def prepare_BP_database(structures):
    clean_database = []
    for ss in structures:
        s = ss
        # del s[[3,4,5,9,10,11]]
        s.positions -= s.positions[np.where(s.arrays['atom_types'] == 0)[0][0]]
        # s.arrays['atom_types'] = np.array([0, 1, 2, 3, 4, 5],dtype=int)
        clean_database.append(s)

    return clean_database

def plot_moire_grid(moire_grid,
                    moire_grid_atoms,
                    moire_grid_gap_vals,
                    moire_structure,
                    plot_title=""):
    
    import matplotlib.patches as patches
    import matplotlib.path as mpath
    
    n_rows = len(moire_grid_atoms)

    fig, ax = plt.subplots(figsize=(10,10))

    vmin = np.array(moire_grid_gap_vals).min()
    vmax = np.array(moire_grid_gap_vals).max()    

    offsets = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
    # offsets = offsets + 1/(2*n)
    offsets = offsets @ moire_structure.cell[:2, :2]

    for i in range(n_rows):
        for j in range(n_rows):
            xy = moire_grid[i][j]

            # Adjust your x0y0 with offsets if needed here
            xy = xy + (offsets * 1.)

            for x0y0 in xy:

                # Define vertices based on your description
                vertices = [
                    (x0y0[0], x0y0[1]),  # Point 1
                    (x0y0[0] + moire_grid_cell[0, 0], x0y0[1] + moire_grid_cell[0, 1]),  # Point 2
                    (x0y0[0] + moire_grid_cell[0, 0] + moire_grid_cell[1, 0],
                    x0y0[1] + moire_grid_cell[0, 1] + moire_grid_cell[1, 1]),  # Point 3
                    (x0y0[0] + moire_grid_cell[1, 0], x0y0[1] + moire_grid_cell[1, 1]),  # Point 4
                    (x0y0[0], x0y0[1]),  # Close the path by returning to the first point
                ]

                # Create a Path object
                codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 2) + [mpath.Path.CLOSEPOLY]
                path = mpath.Path(vertices, codes)

                # Create a PathPatch object
                path_patch = patches.PathPatch(path, facecolor=plt.cm.plasma((moire_grid_gap_vals[i][j] - vmin) / (vmax - vmin)),
                                            edgecolor='none')

                # Add the PathPatch to the Axes
                ax.add_patch(path_patch)


    p1 = np.array([0, 0, 0])
    p2 = moire_structure.cell[0]
    p3 = moire_structure.cell[0] + moire_structure.cell[1]
    p4 = moire_structure.cell[1]
    p_list = np.array([p1, p2, p3, p4])

    # Draw lines around each grid cell
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black')  # Bottom edge
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black')  # Right edge
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color='black')  # Top edge
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], color='black')  # Left edge
    ax.autoscale_view()
    plt.colorbar(plt.cm.ScalarMappable(cmap='plasma',
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                fraction=0.03, ax=ax, label='Relative Bandgap (eV)')
    plt.title(plot_title)
    plt.axis('scaled')
    plot_image_padding = 1.0 * 2 * np.linalg.norm(p3/2)
    plt.xlim(p4[0] - plot_image_padding,
             p2[0] + plot_image_padding)
    plt.ylim(p1[0] - plot_image_padding,
             p3[1] + plot_image_padding)

    plt.savefig("moire_grid_bandgap.png", dpi=300, bbox_inches='tight')


# dataset = read("dataset/mos2_interlayer_dset.xyz", index=":")
dataset = read("dataset/bp_interlayer_dset.xyz", index=":")
# moire_structure = read("structures/relax_546.out", index=-1, format="espresso-out")
# moire_structure = read("structures/scf_1014.out", index=-1, format="espresso-out")
# moire_structure = read("structures/MoS2-2deg-relax.xsf", format="xsf")
moire_atom_types =  read("structures/77-2.15_deg-5472_atoms.xyz", index=-1).arrays['atom_types']
moire_structure = read("structures/relaxed_BL_BP_2.15deg_5472atoms.cif",format = 'cif')
moire_structure.arrays['atom_types'] = moire_atom_types

# moire_structure.arrays['atom_types'] = add_allegro_number_array(moire_structure)

dataset = prepare_BP_database(dataset)

n = 18
moire_grid, moire_grid_atoms = create_moire_grid(moire_structure, n)
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

# n_rows = len(moire_grid_atoms)

# plt.figure(figsize=(10, 10))

# vmin = np.array(moire_grid_gap_vals).min()
# vmax = np.array(moire_grid_gap_vals).max()    

# offsets = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]])
# offsets = offsets + 1/(2*n)
# offsets = offsets @ moire_structure.cell[:2, :2]

# for i in range(n_rows):
#     for j in range(n_rows):
#         x0y0 = moire_grid[i][j]
#         x0y0 = x0y0 + (offsets * 1.)
#         plot1 = plt.scatter(x0y0[:, 0],
#                             x0y0[:, 1],
#                             c=[moire_grid_gap_vals[i][j]]*9,
#                             s=180, marker="h",
#                             vmin=vmin, vmax=vmax,
#                             cmap='plasma')

# p1 = np.array([0, 0])
# p2 = moire_structure.cell[0]
# p3 = moire_structure.cell[0] + moire_structure.cell[1]
# p4 = moire_structure.cell[1]

# # Draw lines around each grid cell
# plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black')  # Bottom edge
# plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black')  # Right edge
# plt.plot([p3[0], p4[0]], [p3[1], p4[1]], color='black')  # Top edge
# plt.plot([p4[0], p1[0]], [p4[1], p1[1]], color='black')  # Left edge
# # plt.xlim(moire_structure.cell[1, 0]-1,
# #          moire_structure.cell[0, 0]+1)
# # plt.ylim(moire_structure.cell[0, 1]-5,
# #          moire_structure.cell[1, 1]+5)
# plt.colorbar(plot1, fraction=0.03,
#                 label='Relative Bandgap (eV)')
# plt.title("Relaxed MoS2/MoS2 bilayer: Relative Bandgap")
# plt.axis('scaled')
# plt.xlim(p4[0] - 5, p2[0] + 5)
# plt.ylim(p1[0] - 5, p3[1] + 5)
# plt.savefig("moire_grid_bandgap.png", dpi=300, bbox_inches='tight')

plot_moire_grid(moire_grid,
                moire_grid_atoms,
                moire_grid_gap_vals,
                moire_structure,
                plot_title="Relaxed BP/BP bilayer: Relative Bandgap")