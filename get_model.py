import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial import distance
import os
from PIL import Image
from itertools import product
import fire
import timm
from timm import create_model
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import_names_dict = {'dino_small':'facebook/dinov2-small',
                     'dino_base':'facebook/dinov2-base',
                     'dino_large':'facebook/dinov2-large',
                     'dino_giant':'facebook/dinov2-giant',

                     'vit_tiny_in22k':'WinKawaks/vit-tiny-patch16-224',
                     'vit_small_in22k':'WinKawaks/vit-small-patch16-224',
                     'vit_base_in22k':"google/vit-base-patch16-224-in21k",
                     'vit_large_in22k':"google/vit-large-patch16-224-in21k",
                     'vit_huge_in22k':"google/vit-huge-patch14-224-in21k",

                     

}

train_set_dict = {## vision transformers
                'vit_small_patch16_224.augreg_in1k':'in1k',
                'vit_base_patch16_224.augreg_in1k':'in1k',
                'vit_small_patch16_224.dino':'in1k',
                'vit_base_patch16_224.dino': 'in1k',

                'vit_tiny_in22k':'in22k',
                'vit_small_in22k':'in22k',
                'vit_base_in22k':"in22k",
                'vit_large_in22k':"in22k",
                'vit_huge_in22k':"in22k",

                'dino_small':'LVD-142M',
                'dino_base':'LVD-142M',
                'dino_large':'LVD-142M',
                'dino_giant':'LVD-142M',

                'vit_base_patch16_clip_224.openai':'openai-400m',
                'vit_large_patch14_clip_224.openai':'openai-400m',

                'vit_base_patch16_clip_224.laion2b':'laion2b',
                'vit_large_patch14_clip_224.laion2b':'laion2b',
                'vit_huge_patch14_clip_224.laion2b':'laion2b',
                'vit_giant_patch14_clip_224.laion2b':'laion2b',




                ## CNNs 
                'convnext_tiny.fb_in1k':'in1k',
                'convnext_small.fb_in1k':'in1k',
                'convnext_base.fb_in1k':'in1k',
                'convnext_large.fb_in1k':'in1k',

                'convnext_tiny.fb_in22k':'in22k',
                'convnext_small.fb_in22k':'in22k',
                'convnext_base.fb_in22k':"in22k",
                'convnext_large.fb_in22k':"in22k",

                'convnext_base.clip_laion2b':'laion2b',
                'convnext_large_mlp.clip_laion2b_augreg':'laion2b',
                'convnext_xxlarge.clip_laion2b_soup':'laion2b',

                'resnet50.a1_in1k':'in1k',
                'resnet50_clip.openai':'openai-400m',
                'resnet18.a1_in1k':'in1k',
                'resnet101.a1_in1k':'in1k',
                'resnet101_clip.openai':'openai-400m',
                
}




def get_model(model_name):

    ######################
    ## Huggingface models
    #####################
    if model_name in import_names_dict.keys():
        import_model_name = import_names_dict[model_name]
        processor = AutoImageProcessor.from_pretrained(import_model_name)
        model = AutoModel.from_pretrained(import_model_name).to(device)

        def net(imgs):
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states = True)
            hidden_layers = outputs.hidden_states
            last_layer = outputs.last_hidden_state
            lst = list(hidden_layers)
            lst.append(last_layer)
            return lst
        

    ######################
    ## Timm models
    #####################
    else:
        model = create_model(model_name, pretrained=True, features_only=True).to(device)    
        model = model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        if ('convnext' or 'resnet') in model_name:
             def net(imgs):
                transformed = list(map(transforms, imgs))
                inputs = torch.stack(transformed).to(device)
                outputs = model(inputs)
                redimmed = [embeddings.reshape([embeddings.shape[0], -1, embeddings.shape[1]]) for embeddings in outputs]
                
                return redimmed
        else:
            def net(imgs):
            # Dictionary to store intermediate outputs
                hidden_states = {}
                # Define a hook function to capture the output of each layer
                def get_features(name):
                    def hook(module, input, output):
                        hidden_states[name] = output.to(device)
                    return hook

                # Register hooks for all layers in the model
                for name, layer in model.named_modules():
                    if 'fc2' in name:
                        print(name)
                        layer.register_forward_hook(get_features(name))
                transformed = list(map(transforms, imgs))
                inputs = torch.stack(transformed).to(device)
                outputs = model(inputs)
                return (list(hidden_states.values()))


        

    # Load and modify the number of parameters file for future plots
    metadata_df = pd.read_csv('utils/metadata_models.csv')
    if model_name not in metadata_df.columns:
        metadata_df[model_name] = [sum(param.numel() for param in model.parameters()), train_set_dict[model_name]]
        metadata_df.to_csv('utils/metadata_models.csv', index=False)
    return net