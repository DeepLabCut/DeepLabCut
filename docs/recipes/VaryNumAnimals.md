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

When labeling, it is always best to label hard frames, i.e. that have all the 6 animals. Labeling videos with less animals and even 1, can still work, but it will be challenging to achieve good individual assembly and tracking.

## In case of existing project

If you already labeled videos with 3 unicorns, but now have videos of the same experimental setting with 6 unicorns, you do not need to retrain the model.

If you run `analyze_videos` you will see that the detections are picked up for all the visible animals. **Before** running `convert_detections2tracklets`, you need to add the additional animals in the config file:

```yaml
individuals:
  # original
  - unicorn 1
  - unicorn 2
  - unicorn 3
  # new
  - unicorn 4
  - unicorn 5
  - unicorn 6
```

and change `topktoretain` in the `project-folder/..path-to-model../test/inference_cfg.yaml` to the max number of animals you want to track, e.g. `topktoretain: 5`.
