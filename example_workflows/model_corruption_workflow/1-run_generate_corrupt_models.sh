#!/bin/bash


# Loop through the seed values and run generate_corrupted_models
for seed in {0..9}
do
    echo "Running generate_corrupted_models with seed $seed"
    python generate_corrputed_models.py --seed "$seed" --modelname lmp_kc_wse2_mos2.pth
done

wait

# Loop through the seed values and run generate_corrupted_models
for seed in {0..9}
do
    echo "Running generate_corrupted_models with seed $seed"
    python generate_corrputed_models.py --seed "$seed" --modelname lmp_sw_wse2.pth
done

wait

for seed in {0..9}
do
    echo "Running generate_corrupted_models with seed $seed"
    python generate_corrputed_models.py --seed "$seed" --modelname lmp_sw_mos2.pth
done

wait