(install-conda)=
# Install with `conda`
DeepLabCut is entirely written in the programming language **Python**, so to run it we need to install Python. There are many ways to do this, but only a few are very robust. Using a virtual environment is a great way to control *which* version of Python we are using, as well as the controlling the versions of all our *dependencies*. This is where `conda` enters the picture.  

```{contents} Contents
---
local:
---
``` 

<!-- Let's make a line to break it up -->
---

## Install `conda`
There are many different versions of `conda`, such as `Anaconda`, `miniconda` and `miniforge`. 
**We recommend using `miniforge` as it is lightweight and fast.**

:::{admonition} Already have `conda` installed?
:class: tip 
If you already have Anaconda installed on your computer, **there's no need to install again!** However, we recommend you set the solver to `libmamba` in the `base` environment to get the same *speed improvement*:
```{code-block} sh
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
:::

::::{tab-set}
:::{tab-item} Windows (GPU)
:sync: win-gpu
[Download and execute the Windows installer](https://github.com/conda-forge/miniforge#download).
Follow the prompts, taking note of the options to "Create start menu shortcuts" and "Add Miniforge3 to my PATH environment variable". The latter is not selected by default due to potential conflicts with other software. Without Miniforge3 on the path, the most convenient way to use the installed software (such as commands conda and mamba) will be via the "Miniforge Prompt" installed to the start menu.
::: 

:::{tab-item} Windows (CPU)
:sync: win-cpu
[Download and execute the Windows installer](https://github.com/conda-forge/miniforge#download).
Follow the prompts, taking note of the options to "Create start menu shortcuts" and "Add Miniforge3 to my PATH environment variable". The latter is not selected by default due to potential conflicts with other software. Without Miniforge3 on the path, the most convenient way to use the installed software (such as commands conda and mamba) will be via the "Miniforge Prompt" installed to the start menu.
::: 

:::{tab-item} MacOS (Intel)
:sync: macos-intel
```{code-block} zsh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
::: 

:::{tab-item} MacOS (M1/2)
:sync: macos-m1
```{code-block} zsh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
::: 

:::{tab-item} Linux
:sync: linux
```{code-block} bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
::: 
::::


## Install GPU prerequisites
You may be using an NVIDIA GPU if you are on either Windows or Linux, or the M1/M2 chip on MacOS for a getting massive speed improvements. If you are on MacOS with M1/M2, then you're in luck - no more work is needed. However, NVIDIA GPUs require a few extra steps... 
::::{tab-set}
:::{tab-item} Windows (GPU)
:sync: win-gpu
- [Install NVIDIA driver](https://www.nvidia.com/Download/index.aspx)
- [Install CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

:::
:::{tab-item} Windows (CPU)
:sync: win-cpu
🚀 No actions needed!
:::

:::{tab-item} MacOS (Intel)
:sync: macos-intel
🚀 No actions needed!
:::

:::{tab-item} MacOS (M1/M2)
:sync: macos-m1
🚀 No actions needed!
:::

:::{tab-item} Linux
:sync: linux
Some actions needed, I can someone help me write this?
:::
::::

## Install DeepLabCut
Now we are all set up to install DeepLabCut. We have made a configuration file which we will use, as it will make this step super easy. 
::::{tab-set}
:::{tab-item} Windows (GPU)
:sync: win-gpu
```{note}
DeepLabCut relies on a deep learning package called `tensorflow`. Since version >2.10 [`tensorflow` no longer supports GPU on Windows](https://www.tensorflow.org/install/source_windows#gpu). The workaround is to manually downgrade our CUDA dependencies, which we do here in the last line - everything should work just fine!
```
```{code-block} pwsh
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut\conda_environments
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
conda install -c=condaforge -c=nvidia cudatoolkit=11.2 cudnn=8.1 cuda-nvcc
```
```{hint}
:tags: [margin]
Here we are first downloading the configuration file (along with the rest of our code), then moving into `conda_environments`. Next, we create our `conda` environment which we subsequently activate.
:::

:::{tab-item} Windows (CPU)
:sync: win-cpu
```{code-block} pwsh
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut\conda_environments
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
```
:::

:::{tab-item} MacOS (Intel)
:sync: macos-intel
```{code-block} zsh
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut/conda_environments
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
```
:::

:::{tab-item} MacOS (M1/M2)
:sync: macos-m1
```{code-block} zsh
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut/conda_environments
conda env create -f DEEPLABCUT_M1.yaml
conda activate DEEPLABCUT
```
:::

:::{tab-item} Linux
:sync: linux
```{code-block} bash
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut/conda_environments
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
```
:::
::::

## Test the installation
To ensure that everything has worked according to plan and that DeepLabCut has successfully installed, we have provided a test script. 
::::{tab-set}
:::{tab-item} All platforms
```{code-block} sh
python testscript_cli.py
```

````{grid}
:gutter: 2

```{grid-item-card} Success!
<!-- :tags: success -->
[**🥳 Congratulations, you're ready to go! You can now use DLC!**](../usage/UseOverviewGuide.md)
```

```{grid-item-card} Something went wrong...
<!-- :tags: error -->
[Have a look at our Installation FAQ](../usage/UseOverviewGuide.md). If that doesn't help, ask a question on the forum. If there's still no luck, you can create an issue on our Github repository.
```
````

:::
::::
<!-- Let's make a line to break it up -->
---

[^driver]: To get the correct drivers, look here.
