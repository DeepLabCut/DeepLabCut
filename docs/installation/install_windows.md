(install-windows)=
# Windows
To use our NVIDIA GPU, we first need to complete a few steps

## 1. Installation prerequisites
::::{tab-set}
:::{tab-item} GPU
:sync: gpu
- Install NVIDIA driver [^driver]
- Install CUDA

:::
:::{tab-item} CPU
:sync: cpu
No actions needed!
:::
::::

## 2. Install DeepLabCut
::::{tab-set}
:::{tab-item} GPU / CPU
:sync: gpu
```{code-block} powershell
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut\conda_environments
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
```
:::
::::

## 3. Install extras
::::{tab-set}
:::{tab-item} GPU
:sync: gpu
```{code-block} powershell 
conda install -c=condaforge -c=nvidia cudatoolkit=11.2 cudnn=8.1 cuda-nvcc
```
:::
:::{tab-item} CPU
:sync: cpu
No actions needed!
:::
::::


## 4. Test that the installation was successful
::::{tab-set}
:::{tab-item} GPU / CPU
:sync: gpu
```{code-block} powershell
python testscript_cli.py
```
:::
::::

## 5. 🥳 Congratulations, you're ready to go! You can now use DLC!
::::{tab-set}
:::{tab-item} GPU / CPU
:sync: gpu
```{code-block} powershell
python -m deeplabcut
```
:::
::::


[^driver]: To get the correct drivers, look here.
