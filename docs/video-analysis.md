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

![Plot poses](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779600836-YOWM5T2MBY0JN1LB537B/plot-poses.png?format=500w)

## Creating a Labeled Video

1. **Go to 'Create Labeled Video' Tab:** The previously analyzed video should be selected.
2. If not already selected, choose your video.
3. Click **`Create Videos`**.

### Viewing the Labeled Video

- Your labeled video will be in your video folder, named after the original video plus model details and 'labeled'.
- Watch the video to assess the model's labeling accuracy.
