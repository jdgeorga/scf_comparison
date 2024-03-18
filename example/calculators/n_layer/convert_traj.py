from ase.io import read, write
from ase.io.trajectory import Trajectory



traj_path = "./octo_BP_relax.traj"
traj = Trajectory(traj_path)
images = []
for atom in traj:
    images.append(atom)

write("octo_BP_traj.xyz", images, format="extxyz")
