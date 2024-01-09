# Installation

### Set up the python environment

```
conda create -n cylindgcn python=3.7
conda activate cylindgcn

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 9.0, install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable
pip install -r requirements.txt

```
### Set up datasets


1. Organize the dataset as the following structure:
    ```
    ├── /path/to/data
    │   ├── train
    │   │   ├── JPGEImages
    │   │   ├── instances_val.json 
    │   ├── val
    │   │   ├── JPGEImages
    │   │   ├── annotations.json 
    ```
2. Create a soft link:
    ```
    ROOT=/path/to/SFEGCN
    cd $ROOT/data
    ln -s /path/to/data data