# Symbolic Language Representation for Zero-Shot Manipulation

This project involves symbolic geometric decomposition for LLM scene understanding, particularly in the domain of grasping.

## Installation

Create the conda environment:
`conda env create -f environment.yml`

Install dependencies:

```
conda install -c conda-forge trimesh
pip install coacd
pip install openai==0.27.9
```
## Getting Started

The pipeline depends on a single-view RGB image and binary mask, and optionally a height image for 3D mode. These files should be named as follows and placed in your specified `data_dir`:

- `{obj}_height.npy` (not needed in 2D mode)
- `{obj}_mask.npy`
- `{obj}_rgb.png`
  
## Running the Demo

The `demo.py` script supports `2d` and `3d` mode. You can specify the mode and the object to process using command-line arguments. Example:

python demo.py --mode 3d --obj knife --data_dir data/


