import numpy as np
from ase import Atoms
from spglib import get_symmetry
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize

class MoirePatternAnalyzer:
    def __init__(self, original_structure, relaxed_structure, energies=None):
        self.original_structure = original_structure
        self.relaxed_structure = relaxed_structure
        self.energies = energies if energies is not None else []

        # Symmetry analysis
        self.sym_data = None

    def calculate_displacement_field(self):
        """Calculate the displacement field between original and relaxed structures."""
        displacement_field = self.relaxed_structure.positions - self.original_structure.positions
        return displacement_field

    def symmetry_analysis(self, symprec=1e-5):
        from scipy.spatial import cKDTree

        """Perform symmetry analysis on the relaxed structure."""
        original_cell = (self.original_structure.cell, self.original_structure.get_scaled_positions(), self.original_structure.numbers)
        sym_data = get_symmetry(original_cell, symprec=symprec)
        rotations = sym_data['rotations']
        translations = sym_data['translations']

        def apply_symmetry_operations(structure, rotations, translations):
            """Apply symmetry operations to the structure and return the transformed positions."""
            symmetry_structures = []
            for rotation, translation in zip(rotations, translations):
                new_structure = structure.copy()
                transformed = np.dot(new_structure.get_scaled_positions(), rotation.T) + translation
                new_structure.set_scaled_positions(transformed)
                # new_structure.wrap()
                symmetry_structures.append(new_structure)
            return symmetry_structures

        def calculate_deviation(structure, transformed_structures):
            """Calculate the deviation between the original and transformed structures."""

            rep_structure = structure.repeat([3,3,1]).copy()
            rep_structure.positions[:,:2] -= (structure.cell[0,:2] + structure.cell[1,:2])

            deviations = []
            for transformed in transformed_structures:
                tree = cKDTree(rep_structure.positions[:,:2])

                deviation, _ = tree.query(transformed.positions[:,:2])
                max_deviation = np.max(deviation)
                deviations.append(max_deviation)

            return deviations

        self.sym_data = sym_data
        transformed_structures = apply_symmetry_operations(self.relaxed_structure,
                                                           rotations,
                                                           translations)
        self.sym_data['max_deviations'] = calculate_deviation(self.relaxed_structure,
                                                              transformed_structures)
        
        return self.sym_data['max_deviations']

    def build_configuration_space(self, layer_1, layer_2, N, interlayer_distance=3):
        cell = layer_1.get_cell()
        translations = [(i/N, j/N) for i in range(N) for j in range(N)]
        self.configuration_space = []
        for t in translations:
            translated_layer_2 = layer_2.copy()
            translated_layer_2.translate(np.dot(t, cell[:2]))
            
            layer_1.positions[:, 2] -= (layer_1.positions[:, 2].max() + interlayer_distance/2.)
            translated_layer_2.positions[:, 2] += layer_1.positions[:, 2].max() - translated_layer_2.positions[:, 2].min() + interlayer_distance


            combined_structure = layer_1 + translated_layer_2
            combined_structure.cell[2,2] = 30.
            # combined_structure.positions[:, 2] = 0
            # combined_structure.set_atomic_numbers(combined_structure.arrays['atom_types'] + 1)
            self.configuration_space.append(combined_structure)
        return self.configuration_space

    def stacking_configuration_space(self, layer_1, layer_2, N, soap_params, data_center_atom_cond):
        """Analyze the stacking configuration space of the relaxed structure."""
        # Step 1: Build configuration space
        self.configuration_space = self.build_configuration_space(layer_1, layer_2, N).copy()

        configurations = []
        for c in self.configuration_space:
            cc = c.copy()
            cc.positions[:, 2] = 0
            cc.set_atomic_numbers(c.arrays['atom_types'] + 1)
            configurations.append(cc)

        flat_original_structure = self.original_structure.copy()
        flat_original_structure.positions[:, 2] = 0
        flat_original_structure.set_atomic_numbers(flat_original_structure.arrays['atom_types'] + 1)

        flat_relaxed_structure = self.relaxed_structure.copy()
        flat_relaxed_structure.positions[:, 2] = 0
        flat_relaxed_structure.set_atomic_numbers(flat_relaxed_structure.arrays['atom_types'] + 1)

        # Step 2: Generate SOAP descriptors
        desc = SOAP(**soap_params)
        config_soap = desc.create(configurations, centers=[[0, 0, 0]] * len(configurations))[:, 0]
        self.config_soap = config_soap
        relaxed_soap = desc.create(flat_relaxed_structure, centers=flat_relaxed_structure.positions[data_center_atom_cond])
        original_soap = desc.create(flat_original_structure, centers=flat_original_structure.positions[data_center_atom_cond])

        interlayer_config_soap = []
        interlayer_relaxed_soap = []
        interlayer_original_soap = []

        for i in range(6):
            for j in range(6):
                if i < 3 and j >= 3:
                    loc_range = desc.get_location([i + 1, j + 1])
                    interlayer_config_soap.append(config_soap[:, loc_range])
                    interlayer_relaxed_soap.append(relaxed_soap[:, loc_range])
                    interlayer_original_soap.append(original_soap[:, loc_range])

        interlayer_config_soap = np.hstack(interlayer_config_soap)
        interlayer_relaxed_soap = np.hstack(interlayer_relaxed_soap)
        interlayer_original_soap = np.hstack(interlayer_original_soap)

        # Normalize SOAP descriptors
        interlayer_config_soap = normalize(interlayer_config_soap)
        interlayer_relaxed_soap = normalize(interlayer_relaxed_soap)
        interlayer_original_soap = normalize(interlayer_original_soap)

        # Step 3: Map local substructures to configuration space using kernel distance
        def map_to_configuration_space(real_soap, config_soap):
            """
            Map local substructures to configuration space using kernel distance.
            args:
                real_soap: SOAP descriptors of the real structure
                config_soap: SOAP descriptors of the configuration space

            returns:
                mapping: Mapping of the local substructures to the configuration space
                         shape (n_config^2, n_real)
            """
            kernel_distances = np.sqrt(2 - 2 * np.tensordot(config_soap, real_soap, axes=([1], [1])))
            inverse_kernel_distances = 1 / (kernel_distances**2)
            mapping = inverse_kernel_distances / inverse_kernel_distances.sum(axis=0)[None, :]
            return mapping

        mapping_original = map_to_configuration_space(interlayer_original_soap, interlayer_config_soap)
        mapping_relaxed = map_to_configuration_space(interlayer_relaxed_soap, interlayer_config_soap)

        self.config_mapping_original = mapping_original
        self.config_mapping_relaxed = mapping_relaxed

        # Step 4: Build histogram
        histogram_original = mapping_original.sum(axis=1).reshape(N, N)
        histogram_relaxed = mapping_relaxed.sum(axis=1).reshape(N, N)

        return histogram_original, histogram_relaxed
    


    def stacking_configuration_space_invariant(self, layer_1, layer_2, N, soap_params, data_center_atom_cond, offset = 0.1):
        """Analyze the stacking configuration space of the relaxed structure."""
        # Step 1: Build configuration space
        self.configuration_space = self.build_configuration_space(layer_1, layer_2, N).copy()

        configurations = []
        for c in self.configuration_space:
            cc = c.copy()
            cc.positions[:, 2] = 0
            cc.set_atomic_numbers(c.arrays['atom_types'] + 1)
            configurations.append(cc)

        flat_original_structure = self.original_structure.copy()
        flat_original_structure.positions[:, 2] = 0
        flat_original_structure.set_atomic_numbers(flat_original_structure.arrays['atom_types'] + 1)
        flat_original_structure_dx = flat_original_structure.copy()
        flat_original_structure_dx.positions[flat_original_structure_dx.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])  
        flat_original_structure_dy = flat_original_structure.copy()
        flat_original_structure_dy.positions[flat_original_structure_dy.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])

        flat_relaxed_structure = self.relaxed_structure.copy()
        flat_relaxed_structure.positions[:, 2] = 0
        flat_relaxed_structure.set_atomic_numbers(flat_relaxed_structure.arrays['atom_types'] + 1)
        flat_relaxed_structure_dx = flat_relaxed_structure.copy()
        flat_relaxed_structure_dx.positions[flat_relaxed_structure_dx.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])
        flat_relaxed_structure_dy = flat_relaxed_structure.copy()
        flat_relaxed_structure_dy.positions[flat_relaxed_structure_dy.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])

        configurations_dx = [c.copy() for c in configurations]
        for c in configurations_dx:
            c.positions[c.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])

        configurations_dy = [c.copy() for c in configurations]
        for c in configurations_dy:
            c.positions[c.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])


        # Step 2: Generate SOAP descriptors
        desc = SOAP(**soap_params)
        config_soap = desc.create(configurations, centers=[[0, 0, 0]] * len(configurations))[:, 0]
        config_soap_dx = desc.create(configurations_dx, centers=[[0, 0, 0]] * len(configurations_dx))[:, 0]
        config_soap_dy = desc.create(configurations_dy, centers=[[0, 0, 0]] * len(configurations_dy))[:, 0]
        self.config_soap = config_soap

        relaxed_soap = desc.create(flat_relaxed_structure, centers=flat_relaxed_structure.positions[data_center_atom_cond])
        relaxed_soap_dx = desc.create(flat_relaxed_structure_dx, centers=flat_relaxed_structure_dx.positions[data_center_atom_cond])
        relaxed_soap_dy = desc.create(flat_relaxed_structure_dy, centers=flat_relaxed_structure_dy.positions[data_center_atom_cond])

        original_soap = desc.create(flat_original_structure, centers=flat_original_structure.positions[data_center_atom_cond])
        original_soap_dx = desc.create(flat_original_structure_dx, centers=flat_original_structure_dx.positions[data_center_atom_cond])
        original_soap_dy = desc.create(flat_original_structure_dy, centers=flat_original_structure_dy.positions[data_center_atom_cond])
        
        interlayer_config_soap = []
        interlayer_config_soap_dx = []
        interlayer_config_soap_dy = []
        interlayer_relaxed_soap = []
        interlayer_relaxed_soap_dx = []
        interlayer_relaxed_soap_dy = []
        interlayer_original_soap = []
        interlayer_original_soap_dx = []
        interlayer_original_soap_dy = []

        for i in range(6):
            for j in range(6):
                if i < 3 and j >= 3:
                    loc_range = desc.get_location([i + 1, j + 1])
                    interlayer_config_soap.append(config_soap[:, loc_range])
                    interlayer_config_soap_dx.append(config_soap_dx[:, loc_range])
                    interlayer_config_soap_dy.append(config_soap_dy[:, loc_range])
                    interlayer_relaxed_soap.append(relaxed_soap[:, loc_range])
                    interlayer_relaxed_soap_dx.append(relaxed_soap_dx[:, loc_range])
                    interlayer_relaxed_soap_dy.append(relaxed_soap_dy[:, loc_range])
                    interlayer_original_soap.append(original_soap[:, loc_range])
                    interlayer_original_soap_dx.append(original_soap_dx[:, loc_range])
                    interlayer_original_soap_dy.append(original_soap_dy[:, loc_range])


        interlayer_config_soap = np.hstack(interlayer_config_soap)
        interlayer_config_soap_dx = np.hstack(interlayer_config_soap_dx)
        interlayer_config_soap_dy = np.hstack(interlayer_config_soap_dy)
        interlayer_relaxed_soap = np.hstack(interlayer_relaxed_soap)
        interlayer_relaxed_soap_dx = np.hstack(interlayer_relaxed_soap_dx)
        interlayer_relaxed_soap_dy = np.hstack(interlayer_relaxed_soap_dy)
        interlayer_original_soap = np.hstack(interlayer_original_soap)
        interlayer_original_soap_dx = np.hstack(interlayer_original_soap_dx)
        interlayer_original_soap_dy = np.hstack(interlayer_original_soap_dy)

        interlayer_config_soap_invariant = np.hstack([interlayer_config_soap,
                                                      interlayer_config_soap_dx,
                                                      interlayer_config_soap_dy])
        interlayer_relaxed_soap_invariant = np.hstack([interlayer_relaxed_soap,
                                                       interlayer_relaxed_soap_dx,
                                                       interlayer_relaxed_soap_dy])
        interlayer_original_soap_invariant = np.hstack([interlayer_original_soap,
                                                        interlayer_original_soap_dx,
                                                        interlayer_original_soap_dy])

        # Normalize SOAP descriptors
        interlayer_config_soap_invariant = normalize(interlayer_config_soap_invariant)
        interlayer_relaxed_soap_invariant = normalize(interlayer_relaxed_soap_invariant)
        interlayer_original_soap_invariant = normalize(interlayer_original_soap_invariant)

        # Step 3: Map local substructures to configuration space using kernel distance
        def map_to_configuration_space(real_soap, config_soap):
            """
            Map local substructures to configuration space using kernel distance.
            args:
                real_soap: SOAP descriptors of the real structure
                config_soap: SOAP descriptors of the configuration space

            returns:
                mapping: Mapping of the local substructures to the configuration space
                         shape (n_config^2, n_real)
            """
            kernel_distances = np.sqrt(2 - 2 * np.tensordot(config_soap, real_soap, axes=([1], [1])))
            inverse_kernel_distances = 1 / (kernel_distances**2)
            mapping = inverse_kernel_distances / inverse_kernel_distances.sum(axis=0)[None, :]
            return mapping

        mapping_original = map_to_configuration_space(interlayer_original_soap_invariant,
                                                      interlayer_config_soap_invariant)
        mapping_relaxed = map_to_configuration_space(interlayer_relaxed_soap_invariant,
                                                     interlayer_config_soap_invariant)

        self.config_mapping_original = mapping_original
        self.config_mapping_relaxed = mapping_relaxed

        # Step 4: Build histogram
        histogram_original = mapping_original.sum(axis=1).reshape(N, N)
        histogram_relaxed = mapping_relaxed.sum(axis=1).reshape(N, N)

        return histogram_original, histogram_relaxed
    
    def stacking_configuration_space_invariant_scale(self, layer_1, layer_2, N, soap_params, data_center_atom_cond, offset = 0.1, scale_thresh = 1e-6):
        """Analyze the stacking configuration space of the relaxed structure."""
        # Step 1: Build configuration space
        self.configuration_space = self.build_configuration_space(layer_1, layer_2, N).copy()

        configurations = []
        for c in self.configuration_space:
            cc = c.copy()
            cc.positions[:, 2] = 0
            cc.set_atomic_numbers(c.arrays['atom_types'] + 1)
            configurations.append(cc)

        flat_original_structure = self.original_structure.copy()
        flat_original_structure.positions[:, 2] = 0
        flat_original_structure.set_atomic_numbers(flat_original_structure.arrays['atom_types'] + 1)
        flat_original_structure_dx = flat_original_structure.copy()
        flat_original_structure_dx.positions[flat_original_structure_dx.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])  
        flat_original_structure_dy = flat_original_structure.copy()
        flat_original_structure_dy.positions[flat_original_structure_dy.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])

        flat_relaxed_structure = self.relaxed_structure.copy()
        flat_relaxed_structure.positions[:, 2] = 0
        flat_relaxed_structure.set_atomic_numbers(flat_relaxed_structure.arrays['atom_types'] + 1)
        flat_relaxed_structure_dx = flat_relaxed_structure.copy()
        flat_relaxed_structure_dx.positions[flat_relaxed_structure_dx.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])
        flat_relaxed_structure_dy = flat_relaxed_structure.copy()
        flat_relaxed_structure_dy.positions[flat_relaxed_structure_dy.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])

        configurations_dx = [c.copy() for c in configurations]
        for c in configurations_dx:
            c.positions[c.arrays['atom_types'] >= 3] += np.array([offset,0.0,0.0])

        configurations_dy = [c.copy() for c in configurations]
        for c in configurations_dy:
            c.positions[c.arrays['atom_types'] >= 3] += np.array([0.0,offset,0.0])


        # Step 2: Generate SOAP descriptors
        desc = SOAP(**soap_params)
        config_soap = desc.create(configurations, centers=[[0, 0, 0]] * len(configurations))[:, 0]
        config_soap_dx = desc.create(configurations_dx, centers=[[0, 0, 0]] * len(configurations_dx))[:, 0]
        config_soap_dy = desc.create(configurations_dy, centers=[[0, 0, 0]] * len(configurations_dy))[:, 0]
        self.config_soap = config_soap

        relaxed_soap = desc.create(flat_relaxed_structure, centers=flat_relaxed_structure.positions[data_center_atom_cond])
        relaxed_soap_dx = desc.create(flat_relaxed_structure_dx, centers=flat_relaxed_structure_dx.positions[data_center_atom_cond])
        relaxed_soap_dy = desc.create(flat_relaxed_structure_dy, centers=flat_relaxed_structure_dy.positions[data_center_atom_cond])

        original_soap = desc.create(flat_original_structure, centers=flat_original_structure.positions[data_center_atom_cond])
        original_soap_dx = desc.create(flat_original_structure_dx, centers=flat_original_structure_dx.positions[data_center_atom_cond])
        original_soap_dy = desc.create(flat_original_structure_dy, centers=flat_original_structure_dy.positions[data_center_atom_cond])
        
        interlayer_config_soap = []
        interlayer_config_soap_dx = []
        interlayer_config_soap_dy = []
        interlayer_relaxed_soap = []
        interlayer_relaxed_soap_dx = []
        interlayer_relaxed_soap_dy = []
        interlayer_original_soap = []
        interlayer_original_soap_dx = []
        interlayer_original_soap_dy = []

        for i in range(6):
            for j in range(6):
                if i < 3 and j >= 3:
                    loc_range = desc.get_location([i + 1, j + 1])
                    interlayer_config_soap.append(config_soap[:, loc_range])
                    interlayer_config_soap_dx.append(config_soap_dx[:, loc_range])
                    interlayer_config_soap_dy.append(config_soap_dy[:, loc_range])
                    interlayer_relaxed_soap.append(relaxed_soap[:, loc_range])
                    interlayer_relaxed_soap_dx.append(relaxed_soap_dx[:, loc_range])
                    interlayer_relaxed_soap_dy.append(relaxed_soap_dy[:, loc_range])
                    interlayer_original_soap.append(original_soap[:, loc_range])
                    interlayer_original_soap_dx.append(original_soap_dx[:, loc_range])
                    interlayer_original_soap_dy.append(original_soap_dy[:, loc_range])


        interlayer_config_soap = np.hstack(interlayer_config_soap)
        interlayer_config_soap_dx = np.hstack(interlayer_config_soap_dx)
        interlayer_config_soap_dy = np.hstack(interlayer_config_soap_dy)
        interlayer_relaxed_soap = np.hstack(interlayer_relaxed_soap)
        interlayer_relaxed_soap_dx = np.hstack(interlayer_relaxed_soap_dx)
        interlayer_relaxed_soap_dy = np.hstack(interlayer_relaxed_soap_dy)
        interlayer_original_soap = np.hstack(interlayer_original_soap)
        interlayer_original_soap_dx = np.hstack(interlayer_original_soap_dx)
        interlayer_original_soap_dy = np.hstack(interlayer_original_soap_dy)

        interlayer_config_soap_invariant = np.hstack([interlayer_config_soap,
                                                      interlayer_config_soap_dx,
                                                      interlayer_config_soap_dy])
        interlayer_relaxed_soap_invariant = np.hstack([interlayer_relaxed_soap,
                                                       interlayer_relaxed_soap_dx,
                                                       interlayer_relaxed_soap_dy])
        interlayer_original_soap_invariant = np.hstack([interlayer_original_soap,
                                                        interlayer_original_soap_dx,
                                                        interlayer_original_soap_dy])

        # Normalize SOAP descriptors
        interlayer_config_soap_invariant = normalize(interlayer_config_soap_invariant)
        interlayer_relaxed_soap_invariant = normalize(interlayer_relaxed_soap_invariant)
        interlayer_original_soap_invariant = normalize(interlayer_original_soap_invariant)

        # Step 3: Map local substructures to configuration space using kernel distance
        def map_to_configuration_space(real_soap, config_soap,scale_thresh):
            """
            Map local substructures to configuration space using kernel distance.
            args:
                real_soap: SOAP descriptors of the real structure
                config_soap: SOAP descriptors of the configuration space

            returns:
                mapping: Mapping of the local substructures to the configuration space
                         shape (n_config^2, n_real)
            """
            kernel_distances = np.sqrt(2 - 2 * np.tensordot(config_soap, real_soap, axes=([1], [1])))
            inverse_kernel_distances = 1 / (kernel_distances**2)
            mapping = inverse_kernel_distances / inverse_kernel_distances.sum(axis=0)[None, :]
            mapping[mapping < scale_thresh] = 0
            mapping /= mapping.sum()

            return mapping

        mapping_original = map_to_configuration_space(interlayer_original_soap_invariant,
                                                      interlayer_config_soap_invariant,
                                                      scale_thresh)
        mapping_relaxed = map_to_configuration_space(interlayer_relaxed_soap_invariant,
                                                     interlayer_config_soap_invariant,
                                                     scale_thresh)

        self.config_mapping_original = mapping_original
        self.config_mapping_relaxed = mapping_relaxed

        # Step 4: Build histogram
        histogram_original = mapping_original.sum(axis=1).reshape(N, N)
        histogram_relaxed = mapping_relaxed.sum(axis=1).reshape(N, N)

        return histogram_original, histogram_relaxed

    def residual_energy(self):
        """Calculate the residual energy between total global energy and sum of local energies."""
        if not self.energies:
            raise ValueError("Energies not provided for residual energy calculation.")

        total_energy = np.sum(self.energies)
        local_energies = np.sum([self.calculate_local_energy(local_structure) for local_structure in self.relaxed_structure])
        residual_energy = total_energy - local_energies
        return residual_energy

    def calculate_local_energy(self, local_structure):
        """Placeholder function to calculate local energy."""
        pass

    def domain_wall_analysis(self, threshold=0.1):
        """Analyze domain walls and calculate their width."""
        displacement_field = self.calculate_displacement_field()
        gradient_norm = np.linalg.norm(np.gradient(displacement_field, axis=0), axis=0)

        def find_significant_region(gradient_norm, threshold):
            significant_region = gradient_norm > threshold
            return significant_region

        def calculate_width(significant_region):
            indices = np.where(significant_region)
            width = np.max(indices) - np.min(indices)
            return width

        significant_region = find_significant_region(gradient_norm, threshold)
        width = calculate_width(significant_region)
        return width

    def fourier_transform_analysis(self):
        """Perform Fourier transform analysis to analyze the periodicity and domain structures."""
        def compute_fourier_transform(structure):
            ft = fftshift(fft2(structure))
            return np.abs(ft)

        original_ft = compute_fourier_transform(self.original_structure)
        relaxed_ft = compute_fourier_transform(self.relaxed_structure)
        return original_ft, relaxed_ft