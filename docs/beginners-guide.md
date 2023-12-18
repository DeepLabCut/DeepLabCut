# Using DeepLabCut

## Installation

Before you begin, make sure that DeepLabCut is installed on your system. For detailed installation instructions, refer to the [Installation Guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html).

## Starting DeepLabCut
>### 🔔 Reminder: How to Open a Terminal
>
> - **Windows:** 
>   - Use the Start menu to search for 'Anaconda Command Prompt' if you are using Miniconda, or 'Command Prompt' if not.
>
> - **Linux:** 
>   - Press `Ctrl + Alt + T` to open a new terminal.
> 
> - **Mac:** 
>   - Press `Cmd + Space` and search for 'Terminal'.

### Activating DeepLabCut Environment

If you have installed DeepLabCut via conda, activate the environment with the following command:

```bash
conda activate DeepLabCut
```
>**⚠️ Attention macOS M1 Users:**
><br/>
>
>🍏 If you are on a macOS with an M1 chip, please use the following special command to activate DeepLabCut:
>```bash
>conda activate DEEPLABCUT_M1

## Launching the DeepLabCut GUI
In the terminal, enter:
```bash
python -m deeplabcut
```
This will open a new DeepLabCut window.

![DeepLabCut GUI Screenshot](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/GUI-screenshot.png?raw=true)

## Starting a New Project

### Navigating the GUI on Initial Launch

When you first launch the GUI, you'll come across three primary main options:

1. **Create New Project:** Geared towards new initiatives. A good choice if you're here to start something new.
2. **Load Project:** Use this to resume your on-hold or past work.
3. **Model Zoo:** Best suited for those who want to explore Model Zoo.

#### Commencing a Your Work:

- For a first-time or new user, please click on **`Start New Project`**.

## Steps to Start a New Project

1. **Launch New Project:**
   - When you start a new project, you'll be presented with an empty project window.

2. **Filling in Project Details:**
   - **Naming Your Project:**
     - Give a specific, well-defined name to your project.
      
      > **💡 Tip:** Avoid empty spaces in your project name.

   - **Naming the Experimenter:**
     - Fill in the name of the experimenter or the project head. This part of the data remains immutable.

3. **Determine Project Location:** 
   - By default, your project will be located on the **Desktop**.
   - To pick a different home, modify the path as needed.

4. **Multi-Animal or Single-Animal Project:**
   - Tick the 'Multi-Animal' option in the menu, but only if that's the mode of the project.
   - Choose the 'Number of Cameras' as per your experiment.

5. **Adding Videos:**
   - First, click on **`Browse Videos`** button on the right side of the window, to search for the video contents.
   - Once the media selection tool opens, navigate and select the folder with your videos.
     
     > **💡 Tip:** DeepLabCut supports **`.mp4`**, **`.avi`**, and **`.mov`** files.
   - A list will be created with all the videos inside this folder.
   - Unselect the videos you wish to remove from the project.
     
6. **Create your project:**
   - Click on **`Create`** button on the bottom, right side of the main window.
   - A new folder named after your project's name will be created in the location you chose above. 
  




