#!/bin/bash

models=(convnext_xxlarge.clip_laion2b_soup)

for model in $models; do
    echo "Running make_RDMs.py for model: $model"
    uv run 1_quadrilaterals/make_RDMs.py --model_name "$model" --method avgdist
done