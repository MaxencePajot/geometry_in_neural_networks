#!/bin/bash

# Extract model names from all_cors.csv (skip header)
models=$(tail -n +2 1_quadrilaterals/all_cors.csv | cut -d',' -f1)

for model in $models; do
    echo "Running make_RDMs.py for model: $model"
    uv run 1_quadrilaterals/make_RDMs.py --model_name "$model" --method avgdist
done