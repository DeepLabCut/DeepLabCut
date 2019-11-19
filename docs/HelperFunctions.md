## Helper Function Documentation

### There are additional functions that are not required, but can be extremely helpful. 

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
```
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
--------

    Converts all pose-output files belonging to mp4 videos in the folder '/media/alex/experimentaldata/cheetahvideos' to csv files. 
    deeplabcut.analyze_videos_converth5_to_csv('/media/alex/experimentaldata/cheetahvideos','.mp4')  
```

While some of the names are ridiculously long, we wanted them to be "self-explanatory." Here is a list 
(that is bound to be continually updated)
of currently available helper functions. To see information about any of them, including HOW 
to use them, use the ``?`` at the end of the call, as described above.


```
deeplabcut.analyze_videos_converth5_to_csv

deeplabcut.mergeandsplit

deeplabcut.analyze_time_lapse_frames

deeplabcut.convertcsv2h5

deeplabcut.ShortenVideo

deeplabcut.DownSampleVideo

deeplabcut.adddatasetstovideolistandviceversa

deeplabcut.comparevideolistsanddatafolders

deeplabcut.dropannotationfileentriesduetodeletedimages
 
deeplabcut.dropduplicatesinannotatinfiles

deeplabcut.load_demo_data

deeplabcut.merge_datasets
```



