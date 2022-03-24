# Using DeepLabCut to track animals with TransformerReID

Sometimes animals cannot be distinguished by a human annotator, but having the animal ID is still desirable. DeepLabCut now uses a Transformer architecture which is trained in an unsupervised setting for tracking animals and inferring ID. For more details on the method, see [Lauer et. al 2022](TODO:AddLinkHere!).

## Code Example
A minimal code example illustrating how to use this tracking-reID method is shown below. This assumes that there is already a multi-animal project created, and frames are extracted and labeled. For more information on how to do these, see the [main multi-animal guide](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html).

```python
import deeplabcut

config = "/home/mammoth/dlc-project/config.yaml"
deeplabcut.train_network(config)

videos_to_analyze = ["/home/mammoth/files/video1.mp4"]

# It is important to set autotrack=False, as this will use the
# normal tracking method by default
deeplabcut.analyze_videos(
    config, videos_to_analyze, videotype="mp4", auto_track=False
)

# Convert the detections to tracklets
deeplabcut.convert_detections2tracklets(
    config, videos_to_analyze, videotype="mp4"
)

# Use the transformer network to track and reID animals
deeplabcut.transformer_reID(
    config, videos_to_analyze, videotype="mp4"
)

# TODO: expose plotting/videocreation of transformer h5 data
```

## Intuition on the Transformer parameters
The function signature is as follows:
```python
def transformer_reID(
    path_config_file,
    videos,
    n_tracks=None,
    train_frac = 0.8, 
    trainingsetindex=0, 
    modelprefix="",
    track_method="ellipse",
    train_epochs=100,
    n_triplets=1000,
    shuffle=1,
)
```

While some arguments are similar to the rest of the DeepLabCut API, `train_epochs` and `n_triplets` are specific to `transformer_reID()`. 

The default values were found to work good in [our paper]((TODO:AddLinkHere!)), however this heavily depends on having good detections beforehand. You should run `deeplabcut.create_video_with_all_detections()` and inspect that the detection performance is sufficient. Another good tool for more advanced users to inspect the quality of assembled animals is `deeplabcut.plot_edge_affinity_distributions()`.

If you find that the detection and assembly performance in a video are sufficient, but the `transformer_reID()` results in poor tracking, you can consider increasing the number of training epochs for the transformer, or the number of triplets (which you can think of as the dataset size that the transformer is training on).





