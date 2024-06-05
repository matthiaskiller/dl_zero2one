# Deep Learning - Zero to One

Repo for getting your DL skills from zero to one!
Starting with the pure basics like how to prepare datasets, build dataloaders, optimizers, solvers and models.
We start from scratch in numpy implementing core DL functionalites which helps use in understanding those concepts. 
In the second stage, we learn how to use DL libraries like PyTorch (Lightning) to build more sophisticated models. 
Examples on Regression, Classification on a variety of tasks (vision, language, etc.)

The repo is partially based on [CS231n Stanford](https://cs231n.stanford.edu) & [I2DL TUM](https://niessner.github.io/I2DL/).

## 1. Python Setup

### Conda setup

`conda create --name dl-zero2one python=3.8`

Next activate the environment using the command:

`conda activate dl-zero2one`

Continue with installation of requirements and starting jupyter notebook as mentioned above, i.e.

`pip install -r requirements.txt` 

`jupyter notebook`


## 2. Installation of PyTorch and Torchlightning

We will use *PyTorch* and *PyTorch Lightning* deep learning frameworks which provide a research oriented interface with a dynamic computation graph and many predefined, learning-specific helper functions.

Since the *PyTorch* installation depends on the individual system configuration (OS, Python version and CUDA version), the desired *PyTorch* package must be installed explicitly (and not from within a `requirements.txt` file).

Use this wheel inside your virtualenv to install *PyTorch*:
### OS X
`pip install torch==1.8.1 torchvision==0.9.1`
### Linux and Windows
```
# CUDA 10.2
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html


# CPU only
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
### Torchlightning

`pip install pytorch-lightning==1.2.8`


## 4. Dataset Download

Datasets will generally be downloaded automatically by notebooks and stored in a common datasets directory shared among all notebooks. A sample directory structure for cifar10 dataset is shown below:-

    dl-zero2one
        ├── datasets                   # The datasets required for all exercises will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 
