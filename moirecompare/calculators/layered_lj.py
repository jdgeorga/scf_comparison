import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.lj import LennardJones, cutoff_function, d_cutoff_function
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
import time


class LayerNeighborList(NeighborList):
    """Optimized filtered neighbor list that includes only certain atom type pairs."""

    def __init__(self, cutoffs, layer_atom_types: list, skin=0.3, sorted=False,
                 self_interaction=True, bothways=False):
        super().__init__(cutoffs, skin, sorted, self_interaction,
                         bothways)
        self.layer1_atom_types = np.array(layer_atom_types[0], dtype=int)
        self.layer2_atom_types = np.array(layer_atom_types[1], dtype=int)

    def update(self, atoms):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.update` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.update`.
        """
        self.atom_types = atoms.arrays['atom_types']
        return self.nl.update(atoms.pbc, atoms.get_cell(complete=True),
                              atoms.positions)

    def get_neighbors(self, atom_index):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors`.
        """
        if self.nl.nupdates <= 0:
            raise RuntimeError('Must call update(atoms) on your neighborlist '
                               'first!')
        
        atom_index_type = self.atom_types[atom_index]

        # Determine atom_index layer
        if atom_index_type in self.layer1_atom_types:
            neighbor_mask = np.isin(self.atom_types[self.nl.neighbors[atom_index]], self.layer2_atom_types)
        elif atom_index_type in self.layer2_atom_types:
            neighbor_mask = np.isin(self.atom_types[self.nl.neighbors[atom_index]], self.layer1_atom_types)
        else:
            raise ValueError(f"Atom index {atom_index} not in layer1 or layer2")
        
        return self.nl.neighbors[atom_index][neighbor_mask], self.nl.displacements[atom_index][neighbor_mask]
    

class LayerLennardJones(LennardJones):
    """Lennard-Jones potential with layer-dependent parameters."""
    implemented_properties = ['energy', 'energies', 'forces', 'free_energy']
    implemented_properties += ['stress', 'stresses']  # bulk properties
    default_parameters = {
        'epsilon': 1.0,
        'sigma': 1.0,
        'rc': None,
        'ro': None,
        'smooth': False,
    }
    nolabel = True

    def __init__(self, layer_atom_types, **kwargs):
        """
        Parameters
        ----------
        sigma: float
          The potential minimum is at  2**(1/6) * sigma, default 1.0
        epsilon: float
          The potential depth, default 1.0
        layer_atom_types: list
            List of lists of atom types for each layer. The first list is for
            the first layer, the second list for the second layer, etc.
        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.
          The energy is upshifted to be continuous at rc.
          Default None
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 0.66 * rc.
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc. Both energy and
          forces are continuous in that case.
          If smooth=True, make sure to check the tail of the
          forces for kinks, ro might have to be adjusted to avoid distorting
          the potential too much.

        """

        super().__init__(**kwargs)
        self.layer_atom_types = layer_atom_types
        if self.layer_atom_types is None:
            raise ValueError("layer_atom_types must be specified")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        t = time.time()
        natoms = len(self.atoms)

        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc
        ro = self.parameters.ro
        smooth = self.parameters.smooth
        layer_atom_types = self.layer_atom_types
        atom_types = self.atoms.arrays['atom_types']
        # if self.nl is None or 'numbers' in system_changes:
        if self.nl is None:
            print("Updating neighbor list")
            self.nl = LayerNeighborList(
                cutoffs=[rc / 2] * natoms,
                layer_atom_types=layer_atom_types,
                self_interaction=False,
                bothways=True
                )
            self.nl.update(self.atoms)
        # print("Time to update neighbor list", time.time() - t)

        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            if atom_types[ii] not in layer_atom_types[0] and atom_types[ii] not in layer_atom_types[1]:
                # print(f"Atom type {atom_types[ii]} not in layer_atom_types")
                continue

            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2

            if smooth:
                cutoff_fn = cutoff_function(r2, rc**2, ro**2)
                d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)

            pairwise_energies = 4 * epsilon * (c12 - c6)
            pairwise_forces = -24 * epsilon * (2 * c12 - c6) / r2  # du_ij

            if smooth:
                # order matters, otherwise the pairwise energy is already
                # modified
                pairwise_forces = (
                    cutoff_fn * pairwise_forces + 2 * d_cutoff_fn
                    * pairwise_energies
                )
                pairwise_energies *= cutoff_fn
            else:
                pairwise_energies -= e0 * (c6 != 0.0)

            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors

            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)

            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product
        # print("Time to loop", time.time() - t)
        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = stresses.sum(
                axis=0) / self.atoms.get_volume()
            self.results['stresses'] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results['energy'] = energy
        self.results['energies'] = energies

        self.results['free_energy'] = energy

        self.results['forces'] = forces
        # print("Time to add results", time.time() - t)



if __name__ == "__main__":
    from ase import Atoms
    from ase.io import read,write

    print("reading atoms")
    atoms = read("bp_interlayer_dset.xyz",index = 0, format = 'extxyz')

    nl = LayerNeighborList(cutoffs=[2.2] * len(atoms),
                        layer_atom_types=[[0,1,2,3],
                                            [4,5,6,7]],
                        skin = 0.001,
                        self_interaction=False,
                        bothways=True)

    nl.update(atoms)
    for ii in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(ii)
        print(f"Atom {ii}:, neighbors: {neighbors}")

    lj = LayerLennardJones(epsilon = 0.0103, sigma = 3.405,
                        layer_atom_types = [[0,1,2,3],[4,5,6,7]],
                        rc = 10.0
                        )
    atoms.calc = lj
    print("Atom energy")
    print(atoms.get_potential_energy())
    print("Atom forces")
    print(atoms.get_forces())