import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count, Value, Lock
from scipy.optimize import minimize
import time
import sys



class StackingConfigurationHistogrammer:
    def __init__(self, structure):
        self.structure = structure
        self.configuration_space = None
        self.interlayer_soap = None

    def build_configuration_space(self, layer_1, layer_2, N, interlayer_distance=3):
        layer_1_cell = layer_1.get_cell()
        layer_2_cell = layer_2.get_cell()
        mean_cell = (layer_1_cell + layer_2_cell) / 2

        cell = mean_cell
        translations = [(i/N, j/N) for i in range(N) for j in range(N)]
        self.configuration_space = []
        for t in translations:
            bottom_layer = layer_1.copy()
            bottom_layer.set_cell(cell, scale_atoms=True)
            translated_layer_2 = layer_2.copy()
            translated_layer_2.translate(np.dot(t, cell[:2]))
            
            layer_1.positions[:, 2] -= (layer_1.positions[:, 2].max() + interlayer_distance/2.)
            translated_layer_2.positions[:, 2] += layer_1.positions[:, 2].max() - translated_layer_2.positions[:, 2].min() + interlayer_distance

            translated_layer_2.set_cell(cell, scale_atoms=True) 
            
            combined_structure = layer_1 + translated_layer_2
            combined_structure.cell[2,2] = 30.
            self.configuration_space.append(combined_structure)
        return self.configuration_space

    def generate_histogram(self, layer_1, layer_2, N, soap_params, data_center_atom_cond, method='kernel', lambda_reg=0.1):
        self.configuration_space = self.build_configuration_space(layer_1, layer_2, N)

        configurations = []
        for c in self.configuration_space:
            cc = c.copy()
            cc.positions[:, 2] = 0
            cc.set_atomic_numbers(c.arrays['atom_types'] + 1)
            configurations.append(cc)

        flat_structure = self.structure.copy()
        flat_structure.positions[:, 2] = 0
        flat_structure.set_atomic_numbers(flat_structure.arrays['atom_types'] + 1)

        print("Calculating SOAP descriptors...")
        desc = SOAP(**soap_params)
        config_soap = desc.create(configurations, centers=[[0, 0, 0]] * len(configurations))[:, 0]
        structure_soap = desc.create(flat_structure, centers=flat_structure.positions[data_center_atom_cond])

        self.interlayer_config_soap = self._extract_interlayer_soap(config_soap, desc)
        self.interlayer_soap = self._extract_interlayer_soap(structure_soap, desc)

        # Normalize SOAP descriptors
        self.interlayer_config_soap = normalize(self.interlayer_config_soap)
        self.interlayer_soap = normalize(self.interlayer_soap)

        if method == 'kernel':
            histogram = self._build_histogram_kernel(self.interlayer_soap, self.interlayer_config_soap, N)
        elif method == 'optimized':
            histogram = self._build_histogram_optimized(self.interlayer_soap, self.interlayer_config_soap, N, lambda_reg)
        else:
            raise ValueError("Invalid method. Choose 'kernel' or 'optimized'.")

        return histogram

    def _extract_interlayer_soap(self, soap, desc):
        interlayer_soap = []
        for i in range(6):
            for j in range(6):
                if i < 3 and j >= 3:
                    loc_range = desc.get_location([i + 1, j + 1])
                    interlayer_soap.append(soap[:, loc_range])
        return np.hstack(interlayer_soap)

    def _build_histogram_kernel(self, real_soap, config_soap, N):
        print("Calculating kernel distances...")
        kernel_distances = np.sqrt(2 - 2 * np.tensordot(config_soap, real_soap, axes=([1], [1])))
        inverse_kernel_distances = 1 / (kernel_distances**2)
        mapping = inverse_kernel_distances / inverse_kernel_distances.sum(axis=0)[None, :]
        return mapping.sum(axis=1).reshape(N, N)

    def _build_histogram_optimized(self, real_soap, config_soap, N, lambda_reg, num_processes=None):
        print("Calculating coefficients...")
        sys.stdout.flush()
        F_T = config_soap.T
        f_R = real_soap.T
        X = self._get_coefficients_parallel(F_T, f_R, lambda_reg, num_processes)
        return X.sum(axis=1).reshape(N, N)

    def _get_coefficients_parallel(self, F_T, f_R, lambda_reg, num_processes=None):
        if num_processes is None:
            num_processes = cpu_count()
        total_tasks = f_R.shape[1]
        print(f"Using {num_processes} processes for {total_tasks} coefficient calculations.")
        sys.stdout.flush()

        # Shared counter for completed tasks
        completed_tasks = Value('i', 0)
        lock = Lock()

        def update_progress(_):
            nonlocal completed_tasks
            with lock:
                completed_tasks.value += 1
                print(f"\rCompleted task {completed_tasks.value}/{total_tasks}", end='')
                sys.stdout.flush()

        start_time = time.time()
        
        with Pool(num_processes) as pool:
            results = []
            for j in range(total_tasks):
                args = (j, F_T, f_R[:, j], lambda_reg)
                res = pool.apply_async(self._get_coefficients_for_j, args=(args,), callback=update_progress)
                results.append(res)
            
            # Wait for all processes to complete
            for res in results:
                res.wait()

        print()  # New line after progress bar
        sys.stdout.flush()

        # Collect results
        X = np.array([res.get() for res in results]).T

        elapsed_time = time.time() - start_time
        print(f"Completed all {total_tasks} tasks - Total elapsed time: {elapsed_time:.2f} seconds")
        sys.stdout.flush()
        
        return X

    @staticmethod
    def _get_coefficients_for_j(args):
        j, F_T, f_R_j, lambda_reg = args
        
        def objective(x, F_T, f_R_j):
            return np.linalg.norm(f_R_j - F_T @ x)**2 + lambda_reg * np.linalg.norm(x)**2

        def constraints(x):
            return [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        bounds = [(0, 1) for _ in range(F_T.shape[1])]

        x0 = np.ones(F_T.shape[1]) / F_T.shape[1]  # Initial guess
        result = minimize(objective, x0, args=(F_T, f_R_j), constraints=constraints(x0), bounds=bounds)
        return result.x

    
def main():
    # Example usage
    structure = read("path_to_structure.xyz")
    
    histogrammer = StackingConfigurationHistogrammer(structure)
    
    # Example parameters (you may need to adjust these)
    layer_1 = structure.copy()  # Assume this is one layer
    layer_2 = structure.copy()  # Assume this is another layer
    N = 12
    soap_params = {
        'species': [1, 2, 3, 4, 5, 6],
        'r_cut': 5.0,
        'n_max': 6,
        'l_max': 6,
        'sigma': 0.05,
        'periodic': True
    }
    data_center_atom_cond = structure.arrays['atom_types'] == 0
    
    # Using kernel method
    histogram_kernel = histogrammer.generate_histogram(
        layer_1, layer_2, N, soap_params, data_center_atom_cond, method='kernel'
    )
    
    # Using optimized method
    histogram_optimized = histogrammer.generate_histogram(
        layer_1, layer_2, N, soap_params, data_center_atom_cond, method='optimized', lambda_reg=0.1
    )
    
    print("Analysis complete. Histogram shapes:")
    print(f"Kernel method: {histogram_kernel.shape}")
    print(f"Optimized method: {histogram_optimized.shape}")

if __name__ == "__main__":
    main()
