# -*- coding: utf-8 -*-
import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
from .michelangelo.utils.misc import instantiate_from_config

def load_surface(fp):
    
    with np.load(fp) as input_pc:
        surface = input_pc['points']
        normal = input_pc['normals']
    
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], 4096, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

def reconstruction(args, model, bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), octree_depth=7, num_chunks=10000):

    surface = load_surface(args.pointcloud_path)
    # old_surface = surface.clone()

    # surface[0,:,0]*=-1
    # surface[0,:,1]*=-1
    surface[0,:,2]*=-1

    # encoding
    shape_embed, shape_latents = model.model.encode_shape_embed(surface, return_latents=True)    
    shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents)

    # decoding
    latents = model.model.shape_model.decode(shape_zq)
    # geometric_func = partial(model.model.shape_model.query_geometry, latents=latents)
    
    return 0

def load_model(ckpt_path="MeshAnything/miche/shapevae-256.ckpt"):
    model_config = OmegaConf.load("MeshAnything/miche/shapevae-256.yaml")
    # print(model_config)
    if hasattr(model_config, "model"):
        model_config = model_config.model

    model = instantiate_from_config(model_config, ckpt_path=ckpt_path)
    model = model.cuda()
    model = model.eval()

    return model
if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, default='./example_data/surface.npz', help='Path to the input point cloud')
    parser.add_argument("--image_path", type=str, help='Path to the input image')
    parser.add_argument("--text", type=str, help='Input text within a format: A 3D model of motorcar; Porsche 911.')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    reconstruction(args, load_model(args))