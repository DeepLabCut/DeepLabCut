# Analysis Tools
   
  Various scripts for analyzing videos with your trained network can be found in the folder ```Analysis-tools```.
   
  After you have a trained a network, you can load the weights to extract poses from novel/other videos. To do so:
   
   - first edit the **myconfig_analysis.py** file 
     
   - For extracting postures, video by video from all videos in a particular folder set the video path in ```myconfig_analysis.py``` accordingly and update the ```Task```, ```date```, ```trainingsFraction```, ```shuffle```, and ```snapshotindex``` variables so that the network of interest is loaded. Then run:
   
    $ CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

in the the ```Analysis-tools``` folder. The postures per frame are stored in a [MultiIndex Panda Arrays](http://pandas.pydata.org/pandas-docs/stable/advanced.html), which contains network and bodypart information as well as the coordinates. These arrays are stored in the efficient [Hierarchical Data Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). The data can also be exported in e.g. ".csv" format or many other formats; see [this discussion](https://github.com/AlexEMG/DeepLabCut/issues/17). 

Once the poses are extracted, one can make labeled videos (```MakingLabeledVideo.py```). This script creates each frame individually and then combines them into a video. This code is slow, but useful for creating flexible high-quality videos (as it has full matplotlib functionality for the individual frames) and allows looking at individual frames. Beyond changing the colorscheme, dotsize, transparency of the markers one can also change the size of the frames in (```myconfig_analysis.py```). Furthermore, one can change the ffmpeg parameters inside (```MakingLabeledVideo.py```). 
   
    $ python3 MakingLabeledVideo.py
      
   There is also a "MakingLabeledVideo_fast.py" script that is much faster, but less flexible. 

     $ python3 MakingLabeledVideo_fast.py

These scripts also illustrate how to load and work with the pose-estimation data. In the future we plan to add more tools. Ideas for analysis scripts and contributions are welcome.

 Return to [readme](../README.md).
