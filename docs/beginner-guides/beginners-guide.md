---
deeplabcut:
  last_content_updated: '2026-03-03'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: outdated
  recommendation: move
  notes: Move to GUI section.
---

(file:beginners-guide)=

# Project Manager GUI - Step by step

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

This guide, and related pages, are meant as a very-new-to-python beginner guide to DeepLabCut. After you are comfortable with this material we recommend then jumping into the more detailed User Guides!

<!-- The course is outdated -->

<!-- - **ProTip:** For even more 'in-depth' understanding, head over to check out the [DeepLabCut Course](https://deeplabcut.github.io/DeepLabCut/docs/course.html), which provides a deeper dive into the science behind DeepLabCut. -->

## Installation

Before you begin, make sure that DeepLabCut is installed on your system.

Please see the {ref}`installation page<file:how-to-install>` for detailed instructions on how to install DeepLabCut on your computer.

<!-- Avoid repeating installation instructions here -->

<!--
## Beginner User Guide

If you are new to Python, the best way to get Python installed onto your computer is with Anaconda. [Head over here and download the version that is best for your computer](https://www.anaconda.com/download).

- "Conda", as it's often called, it a very nice way to create "environments (env)" on your computer. While there can be some cross-talk, in general, it allows you to separate the different tools you need to use to get your science done 💪.

## Let's learn a bit and create a DeeplabCut env:

After you have installed Anaconda, open the new program (Anaconda Terminal). You will be in your "root" directory by default.

**(0) Create a fresh `conda environment`**

In the terminal, type:

```
conda create -n deeplabcut python=3.10
```

You will be prompted (y/n) to install, and then wait for the magic to happen. At the end, check the terminal, it should prompt you to then type:

```
conda activate deeplabcut
```

Now, we are going to install the core dependencies. The way this works is that there are "package managers" such as `conda` itself and python's `pip`. We are going to deploy a mix based on what we know works across ooperating systems.

**(1) Install PyTorch**

`PyTorch` is the backend deep-learning language we wrote DLC3 in. To select the right version, head to the ["Install PyTorch"](https://pytorch.org/get-started/locally/) instructions in the official PyTorch Docs. Select your desired PyTorch build, operating system, select conda as your package manager and Python as the language. Select your compute platform (either a CUDA version or CPU only). Then, use the command to install the PyTorch package. Below are a few possible examples:

- **GPU version of pytorch for CUDA 12.4**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

- **CPU only version of pytorch, using the latest version**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**(2) Install DeepLabCut**

Alright! Next, we will install all the `deeplabcut` source code 🔥. Please decide which version you want (stable or alpha), then type:

- For the **Stable release:**

```
pip install "deeplabcut[gui,modelzoo,wandb]"
```

- This gives you DeepLabCut, the DLC GUI (gui), our latest neural networks (modelzoo) and a cool data logger (wandb) if you choose to use it later on!

- OR for the **Alpha release (from GitHub bleeding edge of the code):**

```
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"
``` -->

## Starting the DeepLabCut GUI

In the terminal, type:

```bash
python -m deeplabcut
```

This will open DeepLabCut.

<!-- (note, the default is dark mode, but you can click "appearance" to change: -->

![DeepLabCut GUI Screenshot](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779625875-5UHPC367I293CBSP8CT6/GUI-screenshot.png?format=500w)

```{note}
For a visual guide on navigating through the DeepLabCut GUI, check out our [YouTube tutorial](https://www.youtube.com/watch?v=tr3npnXWoD4).
```

## Starting a new project

### Navigating the GUI on initial Launch

When you first launch the GUI, you'll find three primary main options:

1. **Create New Project:** Geared towards new initiatives. A good choice if you're here to start something new.
1. **Load Project:** Use this to resume your on-hold or past work.
1. **Model Zoo:** Best suited for those who want to explore Model Zoo.

<!-- ### Creating a : -->

<!-- - For a first-time or new user, please click on **`Start New Project`**. -->

### 🐾 New project step-by-step

1. **Launch New Project:**

   - When you start a new project, you'll be presented with an empty project window. In DLC3+ you will see a new option "Engine".

   ![DeepLabCut Engine](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717780414978-17LOVBUJ8JR102QVSFDY/Screen+Shot+2024-06-07+at+7.13.14+PM.png?format=1500w)

   ```{note}
   For most users, the engine will be PyTorch. See {ref}`sec:deeplabcut-with-tf-install` for TensorFlow support.
   ```

1. **Filling in Project Details:**

   - **Naming Your Project:**

     - Give a specific, easy-to-track name to your project.

     ```{tip}
     Avoid spaces in your project name.
     ```

     - **Fill in the name of the scorer/experimenter**. This name is used in data headers and directory names and it remains permanently associated with the project.

1. **Determine Project Location:**

   - By default, your project will be located on the **Desktop**.
   - To pick a different location, browse as needed.

1. **Multi-Animal or Single-Animal Project:**

   - Tick the 'Multi-Animal' option in the menu if relevant to your experiment.
   - Choose the 'Number of Cameras' as per your experiment.

1. **Adding Videos:**

   - First, click on **`Browse Videos`** button on the right side of the window, to search for the video contents.
   - Once the media selection tool opens, navigate and select the folder with your videos.
     ```{tip}
     DeepLabCut supports **`.mp4`**, **`.avi`**, **`.mkv`** and **`.mov`** files.
     ```
   - A list will be created with all the videos inside this folder.
   - Unselect the videos you wish to remove from the project.
   - Videos outside the project directory can be automatically copied into the project folder by selecting the "Copy videos to project folder" option. This is the recommended strategy for data management. External videos that are not copied are instead referenced via symbolic links. While using symbolic links avoids duplicating files and reduces storage usage, it is also more prone to issues, for example if the original files are moved or deleted.
   - ```{tip}
      By default, the GUI will look for a **directory** containing videos. Use the "Select individual files"
      checkbox if you want to select individual videos instead of a whole folder.
     ```

1. **Define bodyparts and individuals:**

   - Enter all the name, numbers or IDs of bodyparts you wish to track.
     - **Example:** "head", "tail", "left paw", "right paw", etc.
     - Less recommended: "L1", "L2", "L3", etc.
   - **If you have multiple animals**:
     - Enter the name, numbers or IDs of the individuals in your experiment.
       - **Example:** "mouse1", "mouse2", "mouse3", etc.
     - **Unique bodyparts**: If you wish to track "landmark" locations, such as the edges of a maze, or a specific object, you can add these as "unique bodyparts". These are not considered part of an individual, but are still tracked as part of the project.
       - **Example**: "maze_left_edge", "maze_right_edge", "reward_port", etc.
     - **Identity labeling**: if and only if you can tell individuals apart by their appearance (not their location), set this to Yes and consistently label your individuals in the same way across videos. This will allow DeepLabCut to learn to tell them apart, and assign consistent identities across frames and videos.

1. **Create your project:**

   - Click on the **`Create`** button on the bottom, right side of the main window.
   - A new folder will be created in the location you chose above.

## Video tutorial

### 📽 Video Tutorial: Setting Up Your Project in DeepLabCut

![DeepLabCut Create Project GIF](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779616437-30U5RFYV0OY6ACGDG7F4/create-project.gif?format=500w)

## Next steps

Next, head over to the beginner guide for {ref}`editing the configuration and managing the project <file:manage-project-gui>`, which will show you how to edit the configuration file to edit your bodyparts and skeleton structure.
