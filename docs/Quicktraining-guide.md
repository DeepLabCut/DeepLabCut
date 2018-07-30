# Quick Training Guide:

After becoming familiar with the workflow, here is the list of commands to run in the terminal for training. 

You will need to adjust the **myconfig.py** file first (and you will need to adjust your folder names below accordingly):
  
 (A) Place your videos in the folder DeepLabCut-master/Generating_a_Training_Set, first select and label frames:
  
      $ python3 Step1_SelectRandomFrames_fromVideos.py
        
  (B) Then (after labeling all subfolders) create a pandas array with the data:
  
      $ python3 Step2_ConvertingLabels2DataFrame.py
     
  (C) Check labels, and prepare to run: 
  
      $ python3 Step3_CheckLabels.py  #go check the frames in the newly created folders, then if okay, run:
      $ python3 Step4_GenerateTrainingFileFromLabelledData.py
	
 (D) Transfer the folders just created:

       $ cp -R YOURexperimentNameTheDate-trainset95shuffle1 ../pose-tensorflow/models/

       $ cp -R UnaugmentedDataSet_YOURexperimentNameTheDate/ ../pose-tensorflow/models/
 
(E) Download the pre-trained weights (if not already done): 
  
      $ cd ../pose-tensorflow/models/pretrained
      $ ./download.sh
	 
(F) Train: 

      $ cd ../YOURexperimentNameTheDate-trainset95shuffle1/train #change this to your folder name!
      $ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py 

Once your network generalizes well as tested by [evaluation tools](Quickevaluation-guide.md), go to the [analysis tools](analysis-tools.md) to extract poses from videos. For more information on the output of these programs etc., check out the [detailed walk-through with labeled example data](demo-guide.md).

Return to [readme](../README.md).
