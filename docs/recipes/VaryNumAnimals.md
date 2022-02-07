<!-- 

Comments from Mack + Alex:

- having gifs of examples is important here too. 
- also recipes should include GUI use, and links to other steps (ie put a link to more details on “analyze_videos” etc! 


Questions:
----------
- should links direct to code or other tutorials?
- auto_track=True doesn't work for different num of animals right?

-->

# maDLC: Varying the number of animals

In case you have videos of the same experimental setting where the number of animals is changing between videos, DLC can deal with that like a champ. 

## In case of a new project

If you are just now setting up the project, then the most straightforward way to go about this is setting the max number of animals in the config file.

e.g. If there are 2 and up to 6 unicorns per video, then it's best to put them all in the config file. If you can identify them then you can set `identity: true` and:
```yaml
individuals:
  - jonathan
  - samuel
  - miranda
  - lucius
  - vera
  - muriel
```

If you cannot tell the unicorns apart:
```yaml
individuals:
  - unicorn 1
  - unicorn 2
  - unicorn 3
  - unicorn 4
  - unicorn 5
  - unicorn 6
```

When labeling, it is always best to label hard frames, i.e. that have all the 6 animals. Labeling videos with less animals and even 1 works in principle, but practically assembly and tracking performance will be better when training with "hard frames" that have many occlusions and interactions. This is a general recommendation with DeepLabCut, try to label frames that reflect the statistics of luminance, movement & interactions. 

## In case of an existing project

If you already labeled videos with 3 unicorns, but now have videos of the same experimental setting with 6 unicorns you do not need to retrain the model **if you did not train with animal ID**. If you labeled with ID and need to identify the 3 new unicorns too, then you need to train again.

If you run [`analyze_videos`](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#optimized-animal-assembly-video-analysis), and then [create a video with all the detections](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#attention) you will see that the detections are picked up for all the visible animals. In order to get the assemblies and tracks correctly, you need to change `topktoretain` in the `project-folder/..path-to-model../test/inference_cfg.yaml` to the max number of animals you want to track, e.g. `topktoretain: 6`, **before** running [`convert_detections2tracklets`](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#optimized-animal-assembly-video-analysis), 

When running [`stitch_tracklets`](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#optimized-animal-assembly-video-analysis) you will need to pass the correct number of animals in the `n_tracks` argument. 


---

For more details and advice on the complete pipeline of a multi-animal DLC project, check out the [maDLC user guide](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html).