# Video Analysis with DeepLabCut
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">


After training and evaluating your model, the next step is to apply it to your videos.

**How to Analyze Videos**

1. **Navigate to the 'Analyze Videos' Tab:** Begin applying your trained model to video data here.
2. **Select Your Video Format and Files:**
  - **Choose Video Format:** Pick the format of your video (`.mp4`, `.avi`, `.mkv`, or `.mov`).
  - **Select Videos:** Click **`Select Videos`** to find and open your video file.
3. **Start Analysis:** Click **`Analyze`**. The analysis time depends on video length and resolution. Track progress in the terminal or Anaconda prompt.

## Reviewing Analysis Results

- **Find Results in Your Project Folder:** After analysis, go to your project's video folder.
- **Analysis Files:** Look also for a `.metapickle`, an `.h5`, and possibly a `.csv` file for detailed analysis data.
- **Review the "plot-poses" subfolder:** This contains visual outputs of the video analysis.

![Plot poses](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779600836-YOWM5T2MBY0JN1LB537B/plot-poses.png?format=500w)

## Creating a Labeled Video

1. **Go to 'Create Labeled Video' Tab:** The previously analyzed video should be selected.
2. If not already selected, choose your video.
3. Click **`Create Videos`**.

## Viewing the Labeled Video

- Your labeled video will be in your video folder, named after the original video plus model details and 'labeled'.
- Watch the video to assess the model's labeling accuracy.

## Happy DeepLabCutting! 
- Check out the more advanced user guides for even more options!
