import torch
from PIL import Image
from itertools import product
import fire
from scipy.spatial import distance
import pandas as pd
import os
import sys
from pathlib import Path


parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder_path)
import get_model

costs= [1,2,3,4,5,6,7,8,9,10,11,12]

img_costs = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(model_name='dino_small'):

  net = get_model.get_model(model_name)


  images = []
  names = []
  img_costs = []

  for cost in costs:
    for file in os.listdir(f'2_shape_LoT/stimuli/cost-{cost}'):
       if file.endswith('.png') and not file.startswith('.'):
        file_path = os.path.join(f'2_shape_LoT/stimuli/cost-{cost}', file)
        image = Image.open(file_path)
        width, height = image.size 
        new_size = int(width*1.5), int(height*1.5)
        image = image.resize(new_size)
        new_image = Image.new("RGBA", (width, height), "WHITE") # Create a white rgba background
        new_image.paste(image, (-60, -60), image)               # Paste the image on the background. Go to the links given below for details.
        images.append(new_image.convert("RGB"))
        names.append(Path(file_path).stem)
        img_costs.append(cost)
  

  BATCH_SIZE = 16  # Adjust based on memory
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
  if not os.path.exists(f'2_shape_LoT/RDMs/{model_name}/'):
    os.makedirs(f'2_shape_LoT/RDMs/{model_name}/')
    
  ## save RDMs
  for layer in range(len(output)):
    item_embeddings = output[layer]
    item_embeddings = item_embeddings.reshape(item_embeddings.shape[0], -1)

    pairwise_distances = distance.cdist(item_embeddings, item_embeddings)

    distance_dataframe = pd.DataFrame(pairwise_distances)
    distance_dataframe.columns = names
    distance_dataframe.index = names

    # distance_dataframe.to_csv(f'2_shape_LoT/RDMs/{model_name}/layer_{layer}')

if __name__ == "__main__":
    fire.Fire(main)