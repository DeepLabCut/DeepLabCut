
# Automate training and video analysis: Batch Processing

## Tips for working with DLC networks:

Now you have a DLC network and are happy with the performance on selected videos, you may want to run it on all your
videos without hassle. If all your videos are in one folder this is easy, simply pass the foldername to
`deeplabcut.analyze_videos(config,[folder])` and you are fine. What if the videos are scattered?

You can create a simple script that runs over all your video folders with the network of choice. Your "key" to this
network is your config.yaml file.

![](https://static1.squarespace.com/static/57f6d51c9f74566f55ecf271/t/5ccc5abe0d9297405a428522/1556896461304/howtouseDLC-01.png?format=1000w)

Here is a script that you can use to run video analysis over all the folders.

https://github.com/DeepLabCut/DLCutils/tree/master/SCALE_YOUR_ANALYSIS (see below as well)

Note, if a video is analyzed already, it will not be analyzed again! Alternatively, you can push the outputs elsewhere with the flag `destfolder`. See your options by typing: `deeplabcut.analyze_videos?`

Here is an example script. You can copy/paste into a file and end with ".py" to make it a python script.

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:04:37 2019

@author: alex
"""

import os

import deeplabcut

def getsubfolders(folder):
    ''' returns list of subfolders '''
    return [os.path.join(folder, p) for p in os.listdir(folder) if os.path.isdir(os.path.join(folder, p))]

project = "ComplexWheelD3-12-Fumi-2019-01-28"

shuffle = 1

prefix = "/home/alex/DLC-workshopRowland"

projectpath = os.path.join(prefix, project)
config = os.path.join(projectpath, "config.yaml")

basepath = "/home/alex/BenchmarkingExperimentsJan2019"

'''

Imagine that the data (here: videos of 3 different types) are in subfolders:
    /January/January29 ..
    /February/February1
    /February/February2

    etc.

'''

subfolders = getsubfolders(basepath)
for subfolder in subfolders: #this would be January, February etc. in the upper example
    print("Starting analyze data in: ", subfolder)
    subsubfolders = getsubfolders(subfolder)
    for subsubfolder in subsubfolders: #this would be Febuary1, etc. in the upper example...
        print("Starting analyze data in: ", subsubfolder)
        for vtype in [".mp4", ".m4v", ".mpg"]:
            deeplabcut.analyze_videos(config,[subsubfolder],shuffle=shuffle,videotype=vtype,save_as_csv=True)

```

## Now, what about training over multiple Projects

Make your labmates happy by helping run everyone's projects! We use this for workshops, but can easily be adapted for your needs. Here is an example script. You can copy/paste into a file and end with ".py" to make it a python script.
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:12:43 2018

An example script to automate analysis on 3 different GPUs for different projects. Feel free to adapt this to your needs!

@author: alex mathis

"""

import subprocess, sys
import numpy as np
import itertools
import os

import deeplabcut

epochs = 200

model=int(sys.argv[1])

Projects=[["project1-phoenix-2019-01-28"], ["ComplexWheelD3-12-Fumi-2019-01-28", "maze-ariel-2019-01-28"], ["TBI-BvA-2019-01-28", "group-eli-2019-01-28"]]

shuffle=1

prefix = "/home/alex/DLC-workshopRowland"

for project in Projects[model]:
    projectpath = os.path.join(prefix, project)
    config = os.path.join(projectpath, "config.yaml")

    cfg = deeplabcut.auxiliaryfunctions.read_config(config)
    previous_path = cfg["project_path"]

    cfg["project_path"]=projectpath
    deeplabcut.auxiliaryfunctions.write_config(config, cfg)

    print("This is the name of the script: ", sys.argv[0])
    print("Shuffle: ", shuffle)
    print("config: ", config)

    deeplabcut.create_training_dataset(config, Shuffles=[shuffle])

    deeplabcut.train_network(config, shuffle=shuffle, max_snapshots_to_keep=5, epochs=epochs)
    print("Evaluating...")
    deeplabcut.evaluate_network(config, Shuffles=[shuffle], plotting=True)

    print("Analyzing videos..., switching to last snapshot...")
    for vtype in ['.mp4','.m4v','.mpg']:
        try:
            deeplabcut.analyze_videos(config, [str(os.path.join(projectpath, "videos"))], shuffle=shuffle, videotype=vtype, save_as_csv=True)
        except:
            pass

    print("DONE WITH ", project," resetting to original path")
    cfg["project_path"] = previous_path
    deeplabcut.auxiliaryfunctions.write_config(config, cfg)
    ```
