#!/bin/bash
#SBATCH -J relax_corrupt   # Job name
#SBATCH -o relax_corrupt.o%j  # Name of stdout output file
#SBATCH -e relax_corrupt.e%j  # Name of stderr error file
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH -N 1    # Total # of nodes
#SBATCH --ntasks-per-node=128
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH -A m3606           # Project/Allocation name (req'd if you have more than 1)
#SBATCH -c 2
#SBATCH --cpus-per-task=1

ml pytorch/2.0.1


# Define the range of intra and inter corruption values

for seed in {0..9}; do
  for corr_fac in {0..11}; do
    srun --exclusive -N 1 -n 1 --cpus-per-task=4 --cpu-bind=cores python run_corrupt_models_inter_2D.py --seed "$seed" --corr_fac "$corr_fac" &
    sleep 1  # slight delay to stagger the startup times
  done
done
#python corruption_input.py "$file_name" "$intra_corruption" "$inter_corruption" "$num_iterations" &

wait