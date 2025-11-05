# Setting up what keypoints to track
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

**Edit the Configuration File**

After creating your DeepLabCut project, you'll go to the main GUI window, where you'll start managing your project from the Project Management Tab.

**Accessing the Configuration File**

- **Locate the Configuration File:** At the top of the main window, you'll find the file path to the configuration file.
- **Edit the File:** Click on **`Edit config.yaml`**. This action allows you to:
  - Define the bodyparts you wish to track.
  - Outline the skeleton structure (optional!).

A **`Configuration Editor`** window will open, displaying all the configuration details. You'll need to modify some of these settings to align with your research requirements.

## Steps to Edit the Configuration

### 1. Defining Bodyparts

- **Locate the Bodyparts Section:** In the Configuration Editor, find the **`bodyparts`** category.
- **Modify the List:** Click on the arrow next to **`bodyparts`** to expand the list. Here, you can:
  - Update the list with the names of the bodyparts relevant to your study.
  - Add more entries by right-clicking on a row number and selecting **`Insert`**.


![Editing Bodyparts in DeepLabCut's Config File](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779624617-CIVZCM23U69NYK9BO3GY/bodyparts.png?format=500w)


### 2. Defining the Skeleton

- **Navigate to the Skeleton Section:** Scroll down to the **`skeleton`** category.
- **Adjust the Skeleton List:** Click on the arrow to expand this section. You can then:
  - Update the pairs of bodyparts to define the skeleton structure of your model.

![Defining the Skeleton Structure in Config File](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779598505-HQNECHIKSQ6XL033JX8M/skeleton.png?format=500w)

> 💡 **Tip:** If you're new to DeepLabCut, spend some time visualizing how the chosen bodyparts can be connected effectively to form a coherent skeleton.

### Saving Your Changes

- **Save the Configuration:** Once you're satisfied with the modifications, click **`Save`**. This will store your changes and return you to the main GUI window.

## Next, head over the beginner guide for [Labeling your data](labeling)
