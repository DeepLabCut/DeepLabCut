# Quick Anaconda Install for Windows and MacOS!
### Please use one (or more) of the supplied Anaconda environments for a fast, easy install. 

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

If you plant to use Jupyter Notebooks, once you are inside the environment you need to run this line one time to link to Jupyter: ``conda install nb_conda``


Great, that's it! Now just follow the user guide to acvitate your environment and get DeepLabCut up and running in no time!
