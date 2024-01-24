# Labeling

## Selecting Frames to Label

In DeepLabCut, choosing the right frames for labeling is a key step. You've got two easy ways to do this:

1. **Let DeepLabCut Choose:** DeepLabCut can extract frames automatically for you. It's got two neat ways to do that:
   - **Uniform:** This is like taking a snapshot at regular time intervals.
   - **K-means:** This one looks for frames where things are happening differently, giving you a variety of actions and poses.

2. **Pick Frames Yourself:** Just like flipping through a photo album, you can go through your video and pick the frames that catch your eye. Choose the **`manual`** extraction method.

#### Here's how to get started:

- **Step 1:** Click on **`automatic`** in the frame selection area.
- **Step 2:** Choose **`kmeans`** for some variety.
- **Step 3:** Hit the **`Extract Frames`** button, usually found at the bottom right corner.

DeepLabCut will grab 20 frames from each of your videos and put them in a folder named **labeled-data** in your project. Now, you're all set to start labeling!

## Labeling Your First Set of Frames in DeepLabCut

Alright, you've got your extracted frames ready. Now comes the labeling!

### Entering the Label Frames Area

- **Click on `Label Frames`:** This takes you straight to where your frames are, sorted in the **labeled-data** folder, each video in its own subfolder.
- **Open a Folder:** Click on the first one to start, and then click **`open`**.

### The napari DeepLabCut Plugin Scene

- **Plugin Window Opens:** As soon as you click **`open`**, the napari DeepLabCut plugin window appears, your main stage for labeling.
- **Tutorial Popup:** A quick tutorial window shows up. It's a brief guide, so give it a look to understand the basics.

![Labeling Frames in DeepLabCut using Napari Interface](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/labeling-napari.png?raw=true)

### Labeling Setup

- **Frames on Display:** Your frames are lined up in the middle, with a slider below to shuffle through them.
- **Tools and Keypoints:** To the bottom right, you find a list of bodyparts from your configuration. On the top left, all your labeling tools are ready.

### The Labeling Process

- **Start with `Add points`:** Click this to begin placing keypoints on your first frame. If you can't see a bodypart, just move to the next one.
- **Navigate Through Frames:** Use the slider to go from one frame to the next after you're done labeling.
- **Save Progress:** Remember to save your work as you go with **`Command and S`** (or **`Ctrl and S`** on Windows).

> 💡 **Note:** For a detailed walkthrough on using the Napari labeling GUI, have a look at the [DeepLabCut Napari Guide](https://deeplabcut.github.io/DeepLabCut/docs/napari_GUI.html). Additionally, you can watch our instructional [YouTube video](https://www.youtube.com/watch?v=hsA9IB5r73E) for more insights and tips.

![Label your First Frame](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/first-frame.gif?raw=true)

### Completing the Set

Work through all the frames in the first folder. Then, proceed to the next, continuing this way until each folder in your **labeled-data** directory is done. 

## Checking Your Labels

After you've labeled all your frames, it's important to ensure they're accurate. 

### How to Check Your Labels

- **Return to the Main Window:** Once you're done with labeling, head back to DeepLabCut's main window, and click on **`Check Labels`**. 
- **Review the Labeled Folders:** The system will have created new folders for each labeled set inside your labeled-data folder. These folders contain your original frames overlaid with the keypoints you've added.

![Checking Labels in DeepLabCut](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/check-labels.png?raw=true)

Take the time to go through each folder. Accurate labels are key. If there are mistakes, the model might learn incorrectly and mislabel your videos later on. It's all about setting the right foundation for accurate analysis.


















