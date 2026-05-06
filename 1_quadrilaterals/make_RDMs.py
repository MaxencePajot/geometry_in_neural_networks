import torch
from PIL import Image
from itertools import product
import fire
from scipy.spatial import distance
import pandas as pd
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

def main(model_name='dino_small', method='prototype'):

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

        # For each (rotation, scale) instance, embed all 11 shapes and cache the flat embeddings.
        # RDM entries are then the mean pairwise distance across instances.
        embeddings_cache = None  # list of lists: [layer_idx][instance_idx] = (11, features) array

        for d, r in product(range(6), range(6)):
            images = [
                Image.open(f"1_quadrilaterals/stimuli/geom_{shapes[s]}_{types[0]}_{1+d}_{1+r}.png").convert("RGB")
                for s in range(11)
            ]
            with torch.no_grad():
                outputs = net(images)

            if embeddings_cache is None:
                embeddings_cache = [[] for _ in range(len(outputs))]

            for layer_idx, layer_output in enumerate(outputs):
                flat = layer_output.flatten(start_dim=1).detach().cpu().numpy()
                embeddings_cache[layer_idx].append(flat)

        n_instances = 6 * 6

        os.makedirs(f'1_quadrilaterals/RDMs_avgdist/{model_name}/', exist_ok=True)

        for layer_idx in range(len(embeddings_cache)):
            rdm_avg = sum(distance.cdist(flat, flat) for flat in embeddings_cache[layer_idx]) / n_instances
            df = pd.DataFrame(rdm_avg, columns=shapes, index=shapes)
            df.to_csv(f'1_quadrilaterals/RDMs_avgdist/{model_name}/layer_{layer_idx}')

        other_metrics = ['cosine', 'correlation']
        for metric in other_metrics:
            os.makedirs(f'1_quadrilaterals/RDMs_avgdist/z_other_metrics/{metric}/{model_name}/', exist_ok=True)
            for layer_idx in range(len(embeddings_cache)):
                rdm_avg = sum(distance.cdist(flat, flat, metric=metric) for flat in embeddings_cache[layer_idx]) / n_instances
                df = pd.DataFrame(rdm_avg, columns=shapes, index=shapes)
                df.to_csv(f'1_quadrilaterals/RDMs_avgdist/z_other_metrics/{metric}/{model_name}/layer_{layer_idx}')

if __name__ == "__main__":
    fire.Fire(main)
