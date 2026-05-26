import torch
from PIL import Image
from itertools import product
import fire
from scipy.spatial import distance
import pandas as pd
import numpy as np
import os
import sys

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder_path)
import get_model

shapes = [
        "square",
        "rectangle",
        "parallelogram",
        "losange",
        "isoTrapezoid",
        "kite",
        "rightKite",
        "rustedHinge",
        "hinge",
        "trapezoid",
        "random",
    ]
types = ["reference"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_name='dino_small', method='avgdist'):

    net = get_model.get_model(model_name)

    if method == 'prototype':

        prototypes = [[]]*11
        for s in range(11):
            images = []
            # Calculating the embeddings for each shape, with different rotations and sizes
            for d, r in list(product(range(6), range(6))):
                fname = None
                fname = f"1_quadrilaterals/stimuli/geom_{shapes[s]}_{types[0]}_{1+d}_{1+r}.png"
                if fname is not None:
                    img = Image.open(fname).convert("RGB")
                    images.append(img)

            with torch.no_grad():
                outputs = net(images)

            # Calculate the prototype of shape s for each layer
            prototypes[s] = [(lambda x: torch.mean(x, dim=0).flatten().detach().cpu())(x) for x in outputs]


        if not os.path.exists(f'1_quadrilaterals/RDMs/{model_name}/'):
            os.makedirs(f'1_quadrilaterals/RDMs/{model_name}/')

        ## save RDMs

        for layer in range(len(prototypes[0])):

            proto_layer = [sublist[layer] for sublist in prototypes]


            pairwise_distances = distance.cdist(proto_layer, proto_layer)

            distance_dataframe = pd.DataFrame(pairwise_distances)
            distance_dataframe.columns = shapes
            distance_dataframe.index = shapes
            distance_dataframe.to_csv(f'1_quadrilaterals/RDMs/{model_name}/layer_{layer}')


        # same but testing other metrics

        other_metrics = ['cosine', 'correlation']
        for metric in other_metrics:
            ## create folders to save RDMs
            if not os.path.exists(f'1_quadrilaterals/RDMs/z_other_metrics/{metric}/{model_name}/'):
                os.makedirs(f'1_quadrilaterals/RDMs/z_other_metrics/{metric}/{model_name}/')

            ## save RDMs
            for layer in range(len(prototypes[0])):
                proto_layer = [sublist[layer] for sublist in prototypes]


                pairwise_distances = distance.cdist(proto_layer, proto_layer, metric=metric)

                distance_dataframe = pd.DataFrame(pairwise_distances)
                distance_dataframe.columns = shapes
                distance_dataframe.index = shapes

                distance_dataframe.to_csv(f'1_quadrilaterals/RDMs/z_other_metrics/{metric}/{model_name}/layer_{layer}')

    elif method == 'avgdist':

        # Collect all 36 (rotation x scale) embeddings per shape, then compute the mean
        # distance across all 36x36 cross-instance pairs for each shape pair.
        # embeddings_by_shape[s][layer_idx] = (36, features) array
        embeddings_by_shape = [None] * 11

        for s in range(11):
            images = [
                Image.open(f"1_quadrilaterals/stimuli/geom_{shapes[s]}_{types[0]}_{1+d}_{1+r}.png").convert("RGB")
                for d, r in product(range(6), range(6))
            ]
            with torch.no_grad():
                outputs = net(images)
            embeddings_by_shape[s] = [
                layer_output.flatten(start_dim=1).detach().cpu().numpy()
                for layer_output in outputs
            ]

        n_layers = len(embeddings_by_shape[0])

        os.makedirs(f'1_quadrilaterals/RDMs_avgdist/{model_name}/', exist_ok=True)

        all_metrics = ['cosine', 'correlation']
        # all_metrics = [None, 'cosine', 'correlation']

        for metric in all_metrics:
            if metric is None:
                out_dir = f'1_quadrilaterals/RDMs_avgdist/{model_name}/'
            else:
                out_dir = f'1_quadrilaterals/RDMs_avgdist/z_other_metrics/{metric}/{model_name}/'
            os.makedirs(out_dir, exist_ok=True)

            for layer_idx in range(n_layers):
                # Stack all 11*36 embeddings, call cdist once, then average 36x36 blocks
                stacked = np.vstack([embeddings_by_shape[s][layer_idx] for s in range(11)])
                full_dist = distance.cdist(stacked, stacked) if metric is None else distance.cdist(stacked, stacked, metric=metric)
                rdm = np.array([
                    [full_dist[s1*36:(s1+1)*36, s2*36:(s2+1)*36].mean() for s2 in range(11)]
                    for s1 in range(11)
                ])
                df = pd.DataFrame(rdm, columns=shapes, index=shapes)
                df.to_csv(f'{out_dir}layer_{layer_idx}')

if __name__ == "__main__":
    fire.Fire(main)
