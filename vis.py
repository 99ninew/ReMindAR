# python vis.py --subj 1 --autoencoder_name "autoencoder_VAR"


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Reconstructions.ipynb --to python"
# call(command,shell=True)


# In[2]:


import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

# from utils import seed_everything

import utils
from models import Clipper, OpenClipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

# if utils.is_interactive():
#     get_ipython().run_line_magic('load_ext', 'autoreload')
#     get_ipython().run_line_magic('autoreload', '2')

seed=42
# seed_everything(seed=seed)
utils.seed_everything(seed=seed)


# # Configurations

# In[2]:


# # if running this interactively, can specify jupyter_args here for argparser to use
# if utils.is_interactive():
#     # Example use
#     jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
#                     --subj=1 \
#                     --model_name=prior_257_final_subj01_bimixco_softclip_byol"
    
#     jupyter_args = jupyter_args.split()
#     print(jupyter_args)


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="None",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=1,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

# if utils.is_interactive():
    # args = parser.parse_args(jupyter_args)
# else:
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if autoencoder_name=="None":
    autoencoder_name = None

start_scale = 3
# In[4]:

subjj = ''
if subj == 1:
    num_voxels = 15724
    subjj = '01'
elif subj == 2:
    num_voxels = 14278
elif subj == 3:
    num_voxels = 15226
elif subj == 4:
    num_voxels = 13153
elif subj == 5:
    num_voxels = 13039
elif subj == 6:
    num_voxels = 17907
elif subj == 7:
    num_voxels = 12682
elif subj == 8:
    num_voxels = 14386
print("subj",subj,"num_voxels",num_voxels)


# In[5]:
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
val_url = os.path.join(parent_dir, f"dataset/test/test_subj{subjj}_{{0..1}}.tar")
meta_url = os.path.join(parent_dir, f"dataset/metadata_subj{subjj}.json")
# val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
# meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982
batch_size = val_batch_size = 1
voxels_key = 'nsdgeneral.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ## Load autoencoder

# In[6]:


from models import Voxel2StableDiffusionModel
ups_mode = '4x'
outdir = f'../train_logs/models/{autoencoder_name}'
# ckpt_path = os.path.join(outdir, f'epoch048.pth')
ckpt_path = os.path.join(outdir, f'epoch160.pth')

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels, ups_mode=ups_mode, st=start_scale)
    voxel2sd.to(device)
    voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)

    voxel2sd.load_state_dict(state_dict)
    # voxel2sd.load_state_dict(state_dict,strict=False)
    voxel2sd.eval()
    voxel2sd.to(device)
    print("Loaded low-level model!")
else:
    print("No valid path for low-level model specified; not using img2img!") 
    img2img_strength = 1





print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

from VAR.ende import VARImagePipeline
autoencoder = VARImagePipeline(model_depth=36)

retrieve = False
plotting = False
saving = True
verbose = False
imsize = 512


    
ind_include = np.arange(num_val)
all_brain_recons = None
    
only_lowlevel = True
img2img = True
# if img2img_strength == 1:
#     img2img = False
# elif img2img_strength == 0:
#     img2img = True
#     only_lowlevel = True
# else:
#     img2img = True
    
original_images = []
reconstructed_images = []

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl, total=len(ind_include))):
    if val_i < np.min(ind_include):
        continue
    voxel = torch.mean(voxel, axis=1).to(device)
    
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            # ae_preds = [torch.zeros(1, 3, 1, 1).to(device)]
            image_enc_pred_trimmed = [tensor.detach() for tensor in ae_preds]
            blurry_recons = autoencoder.decode(image_enc_pred_trimmed, top_k=900, top_p=0.95, st=2)
            if val_i == 0:
                blurry_recons.save("outputs/result.png")
        else:
            blurry_recons = None

        original_images.append(img.cpu().numpy()) 
        reconstructed_images.append(blurry_recons) 

        if (val_i + 1) % 16 == 0:
            fig, axes = plt.subplots(4, 8, figsize=(16, 8)) 
            for i in range(len(original_images)): 
                row = i // 4
                col = i % 4

                img_np = original_images[i] 
                img_np = img_np.squeeze(0) 
                if img_np.shape[0] == 3: 
                    img_np = img_np.transpose(1, 2, 0)
                axes[row, col].imshow(img_np)
                axes[row, col].axis('off')

                recons_img = reconstructed_images[i]  
                axes[row, col + 4].imshow(np.array(recons_img))  
                axes[row, col + 4].axis('off')

            plt.tight_layout()
            plt.savefig(f"outputs/result1_grid_{val_i // 16}.png")
            plt.close()

            original_images = []
            reconstructed_images = []

        # print(brain_recons.shape)
        # else:
        #     grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
        #         img, voxel,
        #         clip_extractor, unet, vae, noise_scheduler,
        #         voxel2clip_cls = None, #diffusion_prior_cls.voxel2clip,
        #         diffusion_priors = diffusion_priors,
        #         text_token = None,
        #         img_lowlevel = blurry_recons,
        #         num_inference_steps = num_inference_steps,
        #         n_samples_save = batch_size,
        #         recons_per_sample = recons_per_sample,
        #         guidance_scale = guidance_scale,
        #         img2img_strength = img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
        #         timesteps_prior = 100,
        #         seed = seed,
        #         retrieve = retrieve,
        #         plotting = plotting,
        #         img_variations = img_variations,
        #         verbose = verbose,
        #     )

            # if plotting:
            #     plt.show()
            #     # grid.savefig(f'evals/{model_name}_{val_i}.png')

            # brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        # if all_brain_recons is None:
        #     all_brain_recons = brain_recons
        #     all_images = img
        # else:
        #     all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
        #     all_images = torch.vstack((all_images,img))

    if val_i>=np.max(ind_include):
        break

# # print(all_brain_recons)
# all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# if saving:
#     torch.save(all_images,f'all_images.pt')
#     torch.save(all_brain_recons,f'{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples.pt')
# print(f'recon_path: {model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples')

# if not utils.is_interactive():
#     sys.exit(0)
