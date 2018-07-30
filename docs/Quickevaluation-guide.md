# Quick guide for network evaluation:

In the folder **"Evaluation-tools"**, you will find code to evaluate the performance of the trained network on the labeled data set (train and test images).

*For steps that require CUDA, if you are using the [Docker container](https://github.com/AlexEMG/Docker4DeepLabCut), run steps that say ```CUDA_VISIBLE_DEVICES..``` inside the container. We run all the network analysis steps listed below in the container. If you evaluate the nework on a CPU then just ommit the ```CUDA_VISIBLE_DEVICES=0``` command.

**NETWORK EVALUATION:** 

     $ CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py #to evaluate your model [needs TensorFlow <1.2]

     $ python3 Step2_AnalysisofResults.py  #to compute test & train errors for your trained model 

For more information on the output of these programs etc., check out the [detailed walk-through with labeled example data](demo-guide.md).

Return to [readme](../README.md).
