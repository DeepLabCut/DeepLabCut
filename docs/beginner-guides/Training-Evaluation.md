---
deeplabcut:
  last_content_updated: '2025-02-28'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: viable
  recommendation: move
  notes: As mentioned on other beginner-guides/ docs, this should be part of the GUI section.
---

(file:training-evaluation-gui)=

# Neural network training and evaluation in the GUI

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

## Network training

### Creating a training dataset

Before training your model, the first step is to assemble your training dataset.
This involves:

- Splitting labeled data into training and evaluation subsets
- Creating each shuffle folder with the model configuration ready for training.

**Create Training Dataset:** Move to the corresponding tab and click **`Create Training Dataset`**. For starters, the default settings will do just fine. While there are more powerful models and data augmentations you might want to consider, you can trust that for most projects the defaults are a good place to start.

```{note}
This guide assumes you have a (CUDA-enabled) GPU on your local machine. If you're CPU-bound and training is not feasible, consider using Google Colab. Our [Colab Guide](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_YOURDATA_TrainNetwork_VideoAnalysis.ipynb) can help you get started!
```

### Starting the training process

With your training dataset ready, it's time to train your model.

- **Navigate to Train Network:** Head over to the **`Train Network`** tab.
- **Set Training Parameters:** Here, you'll specify:
  - **`Display iterations/epochs`:** To specify how often the training progress will be visually updated. Note that our TensorFlow models are "iterations" while PyTorch is epochs.
  - **`Maximum Iterations/epochs`:** Decide how many iterations to run. For TensorFlow models for a quick demo, 10K is great. For PyTorch models, 200 epochs is fine!
  - **`Number of Snapshots to keep`:** Choose how many snapshots of the model you want to keep, **`Save iterations`:** and at what iteration intervals they should be saved.
- **Launch Training:** Click on **`Train Network`** to begin.

You can keep an eye on the training progress via your terminal window. This will give you a real-time update on how your model is learning (added bonus of the PyTorch model is it also shows you evaluation metrics after each epoch!).

![DeepLabCut Training in Terminal with TF](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779598041-DC8UJA2NXJXG65ZWJH1O/training-terminal.png?format=500w)

## Network evaluation

After training, it's time to see how well your model performs.

### Step-by-step

1. Find and click on the **`Evaluate Network`** tab.
1. **Choose Evaluation Options:**
   - **Plot Predictions:** Select this to visualize the model's predictions, similar to standard DeepLabCut (DLC) evaluations.
   - **Compare Bodyparts:** Opt to compare all the bodyparts for a comprehensive evaluation.
1. Click the **`Evaluate Network`** button, located on the right side of the main window.

```{tip}
If you wish to evaluate all saved snapshots, go to the configuration file and change the `snapshotindex` parameter to `all`.
```

### Interpreting the results

- **Performance Metrics:** DLC will assess the latest snapshot of your model, generating a `.CSV` file with performance
  metrics. This file is stored in the **`evaluation-results`** (for TensorFlow models) or the
  **`evaluation-results-pytorch`** (for PyTorch models) folder within your project.

![Combined Evaluation Results in DeepLabCut](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779617667-0RLTM9DVRALN9YIKSHJZ/combined-evaluation-results.png?format=750w)

- **Visual Feedback:** Additionally, DLC creates subfolders containing your frames overlaid with both the labeled bodyparts and the model's predictions, allowing you to visually gauge the network's performance.

![Evaluation Example in DeepLabCut](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1717779623162-BFDAW37B9TO94EGME2O5/check-labels.png?format=500w)

## Next steps

Head over the {ref}`file:video-analysis-gui` section to learn about applying your trained model to videos, and creating labeled videos with the results of your analysis!
