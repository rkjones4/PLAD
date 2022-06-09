#  PLAD: Learning to Infer Shape Programs with Pseudo-Labels and Approximate Distributions 

By [R. Kenny Jones](https://rkjones4.github.io/), [Homer Walke](https://homerwalke.com/), and [Daniel Ritchie](https://dritchie.github.io/)

![Overview](https://rkjones4.github.io/img/plad/met_plad.png)
 
We present PLAD, a conceptual framework to group a family of related self-supervised learning techniques for shape program inference.
 
## About the paper

Paper: https://rkjones4.github.io/pdf/plad.pdf

Presented at [CVPR 2022](https://cvpr2022.thecvf.com/).

Project Page: https://rkjones4.github.io/plad.html


## Citations
```
@article{jones2022PLAD,
  title={PLAD: Learning to Infer Shape Programs with Pseudo-Labels and Approximate Distributions},
  author={Jones, R. Kenny and Walke, Homer and Ritchie, Daniel},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

# General Information

This repo contains code + data for the experiments presented in our paper. We define 3 shape program domains, and provide instructions for how to integrate additional domains into this framework. For each domain, we provide both PLAD and RL fine-tuning approaches, and also instructions for how to generate a recognition network that has been pretrained on synthetic data.

# Shape Program Domains

**3DCSG**: constructive solid geometry for 3D shapes, by applying boolean operators to 3D primitives. Core definition in [domains/csg3d.py](domains/csg3d.py) .

[**ShapeAssembly**:](https://rkjones4.github.io/shapeAssembly.html): ShapeAssembly is a domain-specific language for specifying the part structure of manufactured 3D objects. It creates objects by declaring cuboid part geometries and assembling those parts together via attachment and symmetry operators. Core definition in [domains/shapeAssembly.py](domains/shapeAssembly.py) . 

**2DCSG**: constructive solid geometry for 3D shapes, by applying boolean operators to 3D primitives. Core definition in [domains/csg2d.py](domains/csg2d.py). To compare against [CSGNet](https://github.com/Hippogriff/CSGNet), we use their language definition. 

# PLAD fine-tuning

The main entrypoint is [main.py](main.py) . This code takes in a program induction domain, and either performs supervised pretraining on synthetic data or fine-tuning against a target distribution of interest.

main.py main arguments:

```
  -dn/--domain_name --> the domain of interest (defined in the domains folder)
  
  -mm/--main_mode  --> what operation main is performing (either fine-tuning or pretraining)
  
  -ts/--train_size --> train set size
  
  -es/--eval_size --> eval set size
  
  -bs/--batch_size --> batch size for training
```

## Example usages

To start a fine-tuning run for the CSG 3D domain with the LEST+ST PLAD mode:

```python3 main.py -dn csg3d -mm fine_tune -ftm LEST_ST```

To start a pretraining run for the CSG 3D domain:

```python3 main.py -dn csg3d -mm pretrain```

## Data

To download the 3D CAD data, please follow the instructions in the BAENet github repo, and use this download [link](https://drive.google.com/file/d/1NvbGIC-XqZGs9pz6wgFwwEPALR-iR8E0/view) . For each of the following categories (chair, table, couch, bench) , please move the {category}/{cateogry_id}\_vox.hdf5 file to data/shape\_net/{category}\_vox.hdf5. Pretrained models for 3D CSG and ShapeAssembly can be download from this [link](https://drive.google.com/file/d/13WoL95yRZOcYmCEBz6rR25kV_zbIeewM/view?usp=sharing), unzip them from the model_output directory.

For 2D CSG data, download the cad.h5 file from this [link](https://www.dropbox.com/s/d6vm7diqfp65kyi/cad.h5?dl=0). Then place this file in data/csgnet_cad/cad.h5 . For 2D CSG, this repo only supports fine-tuning, so first download the pretrained checkpoint from [here](https://www.dropbox.com/s/0f778edn3sjfabp/models.tar.gz?dl=0), and place this file in model_output/CSG2D_SP/models/infer_net.pt . 

## Files

```
infer.py --> Logic that uses the inference network to update the best program data structure
pretrain_sl.py -> Logic for supervised pretraining on a large collection of synthetically generated programs, e.g. samples from the grammar
train_plad.py -> Logic for PLAD fine-tuning
train_rl.py -> Logic for policy gradient fine-tuning, following hyper-parameters from CSGNet wherever possible
utils.py -> Helper functions
```

# Folders

```
data -> where "Real/Taret" shape domains are stored
domains -> where visual program induction (VPU) domains are defined
executors -> associated executor logic for each VPI domain, should also support sampling synthetic programs
model_output -> where outputs from model.py go
models -> model architecture definitions
legacy -> code for 2D CSG, differentiated to integrate wtih the CSGNet codebase.
```

# Adding a New Domain

To run PLAD on a new shape program domain, you will need to complete 4 steps.

1. Define a new domain class
2. Define a new executor
3. Define a inference network
4. Define a target dataset

Two domains, CSG3D and ShapeAssembly, can be used as references.

## Define a new domain

The domain class is the core definition of each shape program domain of interest. Each domain needs to expose the following properties:

What information should get logged during training / evaluation of models (this usually can remain unchanged):

```
TRAIN_LOG_INFO
EVAL_LOG_INFO
```

For **fine-tuning runs**, the following properties need to be implemented:

Argument parsing logic:
```
get_ft_args()
```

How to load the pretrained network for fine-tuning:
```
load_pretrain_net()
```

How to load the dataset of interest to fine-tune on:
```
load_real_data()
```

Logic for initialize RL fine-tuning runs (this usually can remain unchanged):
```
init_rl_run()
```

How to turn the inference network architecture into a generative model (this usually can remain unchanged):
```
create_gen_model()
```

For **pretraining runs**, the following properties need to be implemented:

Argument parsing logic:
```
get_pt_args()
```

Load a recongition network:
```
load_new_net()
```

Load synthetic paired data:
```
get_synth_datasets
```

# Define a new executor

Each domain should have an executor, executor's are responsible for turning programs into shapes. The executor should expose the following properties:

A list of tokens available in the language
```
TOKENS
```

A map of indices to tokens, and a map of tokens to indices:

```
I2T
T2I
```

The expected tensor shape of the visual input:
```
get_input_shape()
```

If get_synth_datasets() is unchanged, then the executor should support the ability to randomly sample a program from the grammar:
```
prog_random_sample()
```

## Define an inference network

The inference network takes an input shape and learns to infer programs that correspond to the input. Each inference network needs to support the following functions:

If create_gen_model() is unchanged, then this function will be called to add a VAE layer in between the encoder and decoder bottleneck.
```
add_vae
```

Convert a batch of visual data (e.g. voxels) into a batch of latent codes with the encoder
```
encode()
```

Used during training, this function takes code, and a target sequence, and runs the target sequence through the network, returning the logit predictions over tokens (see example in model_utils)
```
infer_prog()
```

Logic for inferring programs from input shapes at inference time, in an auto-regressive fashion:
```
eval_infer_progs()
```

Logic for inferring programs from input codes at inference time, in an auto-regressive fashion (should be subset of logic in eval_infer_progs)
```
ws_sample()
```

Logic for forward pass during RL training, see example in model_utils:
```
rl_fwd()
```
## Define a real dataset

The real dataset is the target distribution of shapes of interest; e.g. a distribution of shapes that is known, but lack program annotations (CAD renderings for example). A real dataset should expose the following functions:

Used during RL training, in a while loop, yield batches of visual shapes
```
train_rl_iter()
```

Used during PLAD training, return all training visual shapes
```
get_train_vinput()
```

Used during P Best update step and during evaluation, run inference over shapes using the program inference network:

```
train_eval_iter()
val_eval_iter()
test_eval_iter()
```

# Dependencies

The environment this code was developed in can be found in env.yml - for 2D CSG some additional packages may be required, please see the CSGNet [repo](https://github.com/Hippogriff/CSGNet).
