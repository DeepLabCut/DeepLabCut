# Analysis Tools
   
  Various scripts for analyzing novel videos with your trained network can be found in ```Analysis-tools```.
   
  After you have a trained network one can load the weights to extract poses from novel/other videos. To do so:
   
   - first edit the **myconfig_analysis.py** file 
     
   - For extracting postures video by video from videos in a folder set the video path in ```myconfig_analysis.py```, then run in the the ```Analysis-tools``` folder:
   
    $ CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

The postures per frame are stored in a [MultiIndex Panda Arrays](http://pandas.pydata.org/pandas-docs/stable/advanced.html), which contains network and bodypart information as well as the coordinates. These arrays are stored in the efficient [Hierarchical Data Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). The data can also be exported in e.g. ".csv" format or many other formats; see [this discussion](https://github.com/AlexEMG/DeepLabCut/issues/17). 

Once the poses are extracted, one can make labeled videos (```MakingLabeledVideo.py```). This script creates the video by storing each frame individually and then combining it into a video. This code is slow, but useful for creating flexible high-quality videos (as it has full matplotlib functionality) and looking at individual frames. Beyond changing the colorscheme, dotsize, transparency of the markers one can also change the size of the frames in (```myconfig_analysis.py```). Furthermore, one can change the ffmpeg parameters inside (```MakingLabeledVideo.py```). 
   
    $ python3 MakingLabeledVideo.py
      
   There is also a "MakingLabeledVideo_fast.py" script that is much faster, but less flexible. 

     $ python3 MakingLabeledVideo_fast.py


These scripts also illustrate how to load and work with the pose-estimation data. In the future we plan to add more tools. Ideas for analysis scripts and contribibutions are welcome.

 Return to [readme](../README.md).
