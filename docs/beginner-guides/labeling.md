(labeling)=
# Labeling GUI

## Selecting Frames to Label

In DeepLabCut, choosing the right frames for labeling is a key step. The trick is always to select the MOST DIVERSE data you can that your model will see. That means good lighting, bad lighting, anything you want to throw at it. So, first, pick a range of diverse videos! Then, we will help you pick frames. You've got two easy ways to do this:

1. **Let DeepLabCut Choose:** DeepLabCut can extract frames automatically for you. It's got two neat ways to do that:
   - **Uniform:** This is like taking a snapshot at regular time intervals.
   - **K-means clustering:** This one applies k-means and picks images from different clusters. This is typically better, as it gives you a variety of actions and poses. Note, as it is a clustering tool, it will miss rare events, so ideally run this step, then perhaps consider running the manual GUI to get some rare frames! You can do both within DLC.

2. **Pick Frames Yourself:** Just like flipping through a photo album, you can go through your video and pick the frames that catch your eye - this is great for finding rare frames. Choose the **`manual`** extraction method.

#### Here's how to get started:

- **Step 1:** Click on **`automatic`** in the frame selection area.
- **Step 2:** Choose **`k-means`** for some variety.
- **Step 3:** Hit the **`Extract Frames`** button, usually found at the bottom right corner.

By default, DeepLabCut will grab 20 frames from each of your videos and put them into sub-folders, per video, under **labeled-data** in your project. Now, you're all set to start labeling!

## Labeling Your First Set of Frames in DeepLabCut

Alright, you've got your extracted frames ready. Now comes the labeling!

### Entering the Label Frames Area

- **Click on `Label Frames`:** This takes you straight to where your frames are, sorted in the **labeled-data** folder, each video in its own sub-folder.
- **Open a Folder:** Click on the first one to start, and then click **`open`**.

### The napari DeepLabCut Labeler

- **Plugin Window Opens:** As soon as you click **`open`**, the napari DeepLabCut plugin window appears, your main stage for labeling.
- **Tutorial Popup:** A quick tutorial window shows up. It's a brief guide, so give it a look to understand the basics.

![Labeling Frames in DeepLabCut using Napari Interface](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779602092-LVR2TI6OADSHEYRCGS6F/labeling-napari.png?format=500w))

### Labeling Setup

- **Frames on Display:** Your frames are lined up in the middle, with a slider below to shuffle through them.
- **Tools and Keypoints:** To the bottom right, you find a list of bodyparts from your configuration. On the top left, all your labeling tools are ready.

### The Labeling Process

- **Start with `Add points`:** Click this to begin placing keypoints on your first frame. If you can't see a bodypart, just move to the next one.
- **Navigate Through Frames:** Use the slider to go from one frame to the next after you're done labeling.
- **Save Progress:** Remember to save your work as you go with **`Command and S`** (or **`Ctrl and S`** on Windows).

> 💡 **Note:** For a detailed walkthrough on using the Napari labeling GUI, have a look at the
[DeepLabCut Napari Guide](napari-gui). Additionally, you can watch our instructional
[YouTube video](https://www.youtube.com/watch?v=hsA9IB5r73E) for more insights and tips.


### Completing the Set

Work through all the frames in the first folder. Then, proceed to the next, continuing this way until each folder in your **labeled-data** directory is done. 

## Checking Your Labels

After you've labeled all your frames, it's important to ensure they're accurate. 

### How to Check Your Labels

- **Return to the Main Window:** Once you're done with labeling, head back to DeepLabCut's main window, and click on **`Check Labels`**. 
- **Review the Labeled Folders:** The system will have created new folders for each labeled set inside your labeled-data folder. These folders contain your original frames overlaid with the keypoints you've added.

![Checking Labels in DeepLabCut](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779615252-6BNW661XB2ULH85RTAD3/evaluation-example.png?format=500w)

Take the time to go through each folder. Accurate labels are key. If there are mistakes, the model might learn incorrectly and mislabel your videos later on. It's all about setting the right foundation for accurate analysis.

