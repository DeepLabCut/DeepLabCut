(helper-functions)=
# Helper & Advanced Optional Function Documentation

There are additional functions that are not required, but can be extremely helpful.

First off, if you are new to Python, you might not know this handy trick: you can see
ALL the functions in deeplabcut by typing ``deeplabcut.`` then hitting "tab." You will see a massive list!

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1567907875609-57X4S1LVZWTRJ8GPM34T/ke17ZwdGBToddI8pDm48kKLvSvW2qdCTCjZZgzhLzasUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKc0_818bg8q0aD7_W_W22OLw0yYD6y1fnQ3mVB6beYNdnbXafewWM7FbBaWqQqcLy-/options.png?format=1000w" width="90%">
</p>

Or perhaps you sort of know the name of the function, but not fully, then you can start typing the command, i.e. as in ``deeplabcut.a `` then hit tab:

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1567907844296-STHTZ2SD6UB5WCVEN2I8/ke17ZwdGBToddI8pDm48kJEw9Z-3B5ptjcdSkknf02DlfiSMXz2YNBs8ylwAJx2qgRUppHe6ToX8uSOdETM-XipuQpH02DE1EkoTaghKW779xRs5veesYFcByqmynT9oByNVWkh1tiIAZLs8gRhPycqbSMdPDHKAvTCdk8NbVnE/optionsA.png?format=1000w" width="90%">
</p>


Now, for any of these functions, you type ``deeplabcut.analyze_videos_converth5_to_csv?`` you get:

```python
Signature: deeplabcut.analyze_videos_converth5_to_csv(videopath, videotype='.avi')
Docstring:
By default the output poses (when running analyze_videos) are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position  in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) in the same directory, where the video is stored. If the flag save_as_csv is set to True, the data is also exported as comma-separated value file. However, if the flag was *not* set, then this function allows the conversion of all h5 files to csv files (without having to analyze the videos again)!

This functions converts hdf (h5) files to the comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

 Parameters
----------

    videopath : string
        A strings containing the full paths to videos for analysis or a path to the directory where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.
Only videos with this extension are analyzed. The default is ``.avi``

 Examples
-----------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files.
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')  
```

While some of the names are ridiculously long, we wanted them to be "self-explanatory." Here is a list
(that is bound to be continually updated)
of currently available helper functions. To see information about any of them, including HOW
to use them, use the ``?`` at the end of the call, as described above.


```python
deeplabcut.analyze_videos_converth5_to_csv

deeplabcut.mergeandsplit

deeplabcut.analyze_time_lapse_frames

deeplabcut.convertcsv2h5

deeplabcut.ShortenVideo

deeplabcut.DownSampleVideo

deeplabcut.CropVideo

deeplabcut.adddatasetstovideolistandviceversa

deeplabcut.comparevideolistsanddatafolders

deeplabcut.dropannotationfileentriesduetodeletedimages

deeplabcut.dropduplicatesinannotatinfiles

deeplabcut.load_demo_data

deeplabcut.merge_datasets

deeplabcut.export_model

```

## Model Export function:

This function allows you to export a well-trained single animal model for real-time applications, etc. This function is part of [Kane et al, 2020 eLife](https://elifesciences.org/articles/61909). Please see the paper and related code-base on how to use this utility.

- Another example use case is with the [Bonsai-DeepLabCut](https://github.com/bonsai-rx/deeplabcut) plug in. Namely, you need to first export your trained model from DLC, then follow the instructions for Bonsai-specific use.

```python
deeplabcut.export_model(cfg_path, iteration=None, shuffle=1, trainingsetindex=0, snapshotindex=None, TFGPUinference=True, overwrite=False, make_tar=True)
```

## Advanced Labeling across Cameras:

### If you have two cameras and you want to make a 3D Project from your data, you can leverage this in the Labeling GUI:

If you have multiple cameras, you may want to use epipolar lines projected on the images you are labeling to help you label the same position on the body in each camera angle. An epipolar line is a projection from one camera to all the possible points in the second camera's image that could match the labeled point in the first camera's image. A correctly labeled point will fall somewhere along this projected line.

In order to label with epipolar lines, you must complete two additional sets of steps **prior to labeling.**

- First, you must create a 3d project and calibrate the cameras - to do so, complete
  steps 1-3 in [3D Overview](3D-overview).

- Second, you must extract imagr from `camera_1` first; here you would have run the standard `deeplabcut.extract_frames(config_path, userfeedback=True)`, but just extract files from 1 camera. Next, you need to extract matching frames from `camera_2`:
```python
deeplabcut.extract_frames(config_path, mode = 'match', config3d=config_path3d, extracted_cam=0)
```
You can set `extracted_cam=0` to match all other camera images to the frame numbers in the `camera_1` folder, or change this to match to other cameras. If you `deeplabcut.extract_frames` with `mode='automatic'` before, it shouldn't matter which camera you pick. If you already extracted from both cameras, be warned this will overwrite the images for `camera_2`.

- Three, now you can label with epipolar lines:

     - Here, label `camera_1` as you would normally, i.e.:
    ```python
    deeplabcut.label_frames(config_path)
    ```
    - Then for `camera_2` (now it will compute the epipolar lines based on camera_1 labels and project them onto the GUI):
    ```python
    deeplabcut.label_frames(config_path, config3d=config_path3d)
    ```
