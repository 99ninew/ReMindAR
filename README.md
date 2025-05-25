# ReMindAR

Reconstruct Mind Autoregressively

## Installation instructions

1. Agree to the Natural Scenes Dataset's [Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) and fill out the [NSD Data Access form](https://forms.gle/xue2bCdM9LaFNMeb7)
2. Download this repository: `git clone https://github.com/99ninew/ReMindAR.git`
3. Run `set.up` to create a conda environment that contains all the necessary packages required  to run our codes. Then, activate the environment with `conda activate remindar`

```cmd
cd src
. setup.sh
```

## General information

This repository contains Python files and Jupytor notebooks for

1. Defining the VAR model (src/VAR)
2. Training ReMindAR's VAR pipeline and obtaining initial reconstructions from brain activity (src/train_with_var.py)
3. Training ReMindAR's CLIP pipeline (src/train_with_clip.py)
4. Reconstructing images from fMRI data using the trained model (src/Reconstructions.ipynb)
5. Evaluating reconstructions against the ground truth stimuli using various low-level and high-level metrics (src/Reconstruction_Metrics.ipynb)

Besides, all the above Jupytor notebooks have corresponding python files. 

## Honor Code

We refer to the high-level pipeline training and evaluation methods outlined in the [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) Github repository.