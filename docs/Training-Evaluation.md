# Training and Evaluation

## Preparing Your Training Dataset

Before training your model, the first step is to assemble your training dataset.

- **Create Training Dataset:** Move to the corresponding tab and click **`Create Training Dataset`**. For starters, the default settings will do just fine. You can delve into the specifics of each option later on.

> 💡 **Note:** This guide assumes you have a GPU on your local machine. If you're CPU-bound and finding training challenging, consider using Google Colab. Our [Colab Guide](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_YOURDATA_TrainNetwork_VideoAnalysis.ipynb) can help you get started!

## Kickstarting the Training Process

With your training dataset ready, it's time to train your model.

- **Navigate to Train Network:** Head over to the **`Train Network`** tab.
- **Set Training Parameters:** Here, you'll specify:
  - **`Display iterations`:** To specify how often the training progress will be visually updated.
  - **`Maximum Iterations`:** Decide how many iterations to run. For this tutorial, let's set it to 100,000.
  - **`Number of Snapshots to keep`:** Choose how many snapshots of the model you want to keep, **`Save iterations`:** and at what iteration intervals they should be saved.
- **Launch Training:** Click on **`Train Network`** to begin.

You can keep an eye on the training progress via your terminal window. This will give you a real-time update on how your model is learning.

![DeepLabCut Training in Terminal](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/training-terminal.png?raw=true)

## Evaluate the Network

After training, it's time to see how well your model performs.

### Steps to Evaluate the Network

1. Find and click on the **`Evaluate Network`** tab.
2. **Choose Evaluation Options:**
   - **Plot Predictions:** Select this to visualize the model's predictions, similar to standard DeepLabCut (DLC) evaluations.
   - **Compare Bodyparts:** Opt to compare all the bodyparts for a comprehensive evaluation.
3. Click the **`Evaluate Network`** button, located on the right side of the main window.

>💡 Tip: If you wish to evaluate all saved snapshots, go to the configuration file and change the `snapshotindex` parameter to `all`. 


### Understanding the Evaluation Results

- **Performance Metrics:** DLC will assess the latest snapshot of your model, generating a `.CSV` file with performance metrics. This file is stored in the **`evaluate network`** folder within your project.


![Combined Evaluation Results in DeepLabCut](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/combined-evaluation-results.png?raw=true)
- **Visual Feedback:** Additionally, DLC creates subfolders containing your frames overlaid with both the labeled bodyparts and the model's predictions, allowing you to visually gauge the network's performance.

![Evaluation Example in DeepLabCut](https://github.com/Timokleia/DeepLabCut/blob/create-tutorial/docs/images/evaluation-example.png?raw=true)
















