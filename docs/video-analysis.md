# Video Analysis with DeepLabCut

After training and evaluating your model, the next step is to apply it to your videos.

## How to Analyze Videos

1. **Navigate to the 'Analyze Videos' Tab:** Begin applying your trained model to video data here.
2. **Select Your Video Format and Files:**
  - **Choose Video Format:** Pick the format of your video (`.mp4`, `.avi`, or `.mov`). 
  - **Select Videos:** Click **`Select Videos`** to find and open your video file.
3. **Start Analysis:** Click **`Analyze`**. The analysis time depends on video length and resolution. Track progress in the terminal or Anaconda prompt.

### Reviewing Analysis Results

- **Find Results in Your Project Folder:** After analysis, go to your project's video folder.
- **Analysis Files:** Look also for a `.metapickle`, an `.h5`, and possibly a `.csv` file for detailed analysis data.
- **Review the Plot Poses Subfolder:** This contains visual outputs of the video analysis.

<p align="center">
  <img src="https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/plot-poses.png?raw=true" alt="Plot Poses in DeepLabCut">
</p>

## Creating a Labeled Video

1. **Go to 'Create Labeled Video' Tab:** The previously analyzed video should be selected.
2. If not already selected, choose your video.
3. Click **`Create Videos`**.

### Viewing the Labeled Video

- Your labeled video will be in your video folder, named after the original video plus model details and 'labeled'.
- Watch the video to assess the model's labeling accuracy.

<p align="center">
  <img src="https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/labeled-video.gif?raw=true" alt="Labeled Video in DeepLabCut">
</p>



