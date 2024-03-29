from ase.io import read,write
import numpy as np
from moirecompare.calculators import AllegroCalculator, NLayerCalculator
from moirecompare.calculators import (MonolayerLammpsCalculator, 
                                      BilayerLammpsCalculator,
                                      InterlayerLammpsCalculator)
from ase.optimize import FIRE, BFGS


atoms = read("./1D_25cells_0deg.xyz", format="extxyz")
atoms.positions[:,2] += 15
atoms.cell[2,2] = 100
print(atoms.positions[:,2])

at_1 = atoms.copy()[atoms.arrays["atom_types"] < 3]
at_2 = atoms.copy()[atoms.arrays["atom_types"] >= 3]


layer_symbols = [["Mo", "S", "S"],
                 ["Mo", "S", "S"]]

intralayer_calcs = [
      MonolayerLammpsCalculator(atoms[atoms.arrays['atom_types'] < 3],
                                layer_symbols[0],
                                system_type='TMD',
                                intra_potential='tmd.sw')]
interlayer_calcs = []

for i in np.arange(1, len(layer_symbols)):
      layer_atoms = atoms[
            np.logical_and(atoms.arrays['atom_types'] >= i*3,
                           atoms.arrays['atom_types'] < (i+1)*3)
                           ]
      print(np.unique(layer_atoms.arrays['atom_types']))
      intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms,
                                                layer_symbols=layer_symbols[i],
                                                system_type='TMD',
                                                intra_potential = 'tmd.sw'))
      
      bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i-1)*3,
                                         atoms.arrays['atom_types'] < (i+1)*3)]
      print(np.unique(bilayer_atoms.arrays['atom_types']))
      print(layer_symbols[i-1:i+1])

      interlayer_calcs.append(
            InterlayerLammpsCalculator(bilayer_atoms,
                                       layer_symbols=layer_symbols[i-1:i+1],
                                       system_type='TMD'))


n_layer_calc = NLayerCalculator(atoms,
                                intralayer_calcs,
                                interlayer_calcs,
                                layer_symbols)


# bilayer_lammps = BilayerLammpsCalculator(atoms,
#                                          chemical_symbols=[["Mo", "S", "S"],
#                                                             ["Mo", "S", "S"]],
#                                            system_type='TMD')
                                         

atoms.calc = n_layer_calc 

atoms.calc.calculate(atoms)

print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
      f"layer_energy {atoms.calc.results['layer_energy']}")

dyn = FIRE(atoms,trajectory = '1D_MoS2_0deg_relax_25cells_FIRE_nlayer_lammps.traj')
dyn.run(fmax=5e-3)

# atoms.calc.minimize(atoms, method = 'fire')

print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
      f"layer_energy {atoms.calc.results['layer_energy']}")


from ase.io import read, write
from ase.io.trajectory import Trajectory

traj_path = "./1D_MoS2_0deg_relax_25cells_FIRE_nlayer_lammps.traj"
traj = Trajectory(traj_path)
images = []
for atom in traj:
    images.append(atom)

write("1D_MoS2_0deg_relax_25cells_FIRE_nlayer_lammps_traj.xyz", images, format="extxyz")
