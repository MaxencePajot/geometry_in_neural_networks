import torch
from PIL import Image
from itertools import product
import fire
from scipy.spatial import distance
import pandas as pd
import os
import numpy as np
import sys
import itertools

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder_path)
import get_model

renderings= ['photo', 'drawing', 'geo', 'word']
items = ['face', 'profil','eyes', 'face2', 'walking', 'standing', 'hand', 'legs', 'bird', 'butterfly', 'girafe', 'fish', 'cherry', 'tree', 'flower', 'carrot', 'watch', 'key', 'truck', 'velo', 'road', 'house', 'mountain', 'windmill']
dico = {}
count = 1
rendering_items = []
for rendering in renderings:
  for item in items:
    rendering_items.append(f'{rendering}_{item}')
    dico[f'{rendering}_{item}'] = count
    count += 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_name='dino_small'):

    net = get_model.get_model(model_name)


    images = []
    for rendering in renderings:
        for item in items:
            key = f"{rendering}_{item}"
            images.append(Image.open(f'3_abstract_drawings/stimuli/{dico[key]}.png').convert("RGB"))
    BATCH_SIZE = 16  

    all_outputs = None  # To store layer-wise outputs

    # Process images in batches
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]  # Get batch slice
        with torch.no_grad():
            batch_outputs = net(batch)  # Process batch (list of tensors)

            # Initialize storage for outputs (if first batch)
            if all_outputs is None:
                all_outputs = [[] for _ in range(len(batch_outputs))]

            # Append each layer's output separately
            for j, layer_output in enumerate(batch_outputs):
                all_outputs[j].append(layer_output)

    # Concatenate outputs for each layer separately
    output = [torch.cat(layer_outputs, dim=0).cpu() for layer_outputs in all_outputs]

    ## create folders to save RDMs
    if not os.path.exists(f'3_abstract_drawings/RDMs/{model_name}/'):
        os.makedirs(f'3_abstract_drawings/RDMs/{model_name}/')
    ## save RDMs
    for layer in range(len(output)):
        item_embeddings = output[layer]
        item_embeddings = item_embeddings.reshape(item_embeddings.shape[0], -1)

        pairwise_distances = distance.cdist(item_embeddings, item_embeddings)

        distance_dataframe = pd.DataFrame(pairwise_distances)
        distance_dataframe.columns = rendering_items
        distance_dataframe.index = rendering_items

        distance_dataframe.to_csv(f'3_abstract_drawings/RDMs/{model_name}/layer_{layer}')

if __name__ == "__main__":
    fire.Fire(main)