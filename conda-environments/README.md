# Quick Anaconda Install for Windows and MacOS!
### Please use one (or more) of the supplied Anaconda environments for a fast and easy installation process.

(0) Be sure you have Anaconda 3 installed! https://www.anaconda.com/distribution/, and get familiar with using "cmd" or terminal!

(1) download or git clone this repo (in the terminal/cmd program, while in a folder you wish to place DeepLabCut 
type ``git clone https://github.com/AlexEMG/DeepLabCut.git``

(2) "cd", i.e. go into, the folder named ``conda-environments``

(2) Now, depending on which file you want to use, in your terminal type: 

``conda env create -f dlc-macOS-CPU.yaml``

or 

``conda env create -f dlc-windowsCPU.yaml``

or 

``conda env create -f dlc-windowsGPU.yaml``

If you plan to use Jupyter Notebooks once you are inside the environment you need to run this line one time to link to Jupyter: ``conda install nb_conda``


Great, that's it! 

Now just follow the user guide, to activate your environment and get DeepLabCut up and running in no time!

Just as a reminder, you can exit the environment anytime and (later) come back! So the environments really allow you to manage multiple packages that you might want to install on your computer. 

Once you are in the terminal type:
- Windows: ``activate nameoftheenvironment``
- Linux/MacOS: ``source activate nameoftheenvironment``

Here are some conda environment management tips: https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index

