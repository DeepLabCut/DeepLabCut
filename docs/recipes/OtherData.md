# How to use data labeled outside of DeepLabCut
- and/or if you merge projects across scorers (see below):



## Using data labeled elsewhere:

Some users may have annotation data in different formats, yet want to use the DLC pipeline. In this case, you need to convert the data to our format. Simply, you can format your data in an excel sheet (.csv file) or pandas array (.h5 file).

Here is a guide to do this via the ".csv" route: (the pandas array route is identical, just format the pandas array in the same way). 

**Step 1**: create a project as describe in the user guide: https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/UseOverviewGuide.md#create-a-new-project

**Step 2**: edit the ``config.yaml`` file to include the body part names, please take care that spelling, spacing, and capitalization are IDENTICAL to the "labeled data body part names". 

**Step 3**: Please inspect the excel formatted sheet (.csv) from our [demo project](https://github.com/DeepLabCut/DeepLabCut/tree/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1)
- i.e. this file: https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.csv

**Step 4**: Edit the .csv file such that it contains the X, Y pixel coordinates, the body part names, the scorer name as well as the relative path to the image: e.g. /labeled-data/somefolder/img017.jpg 
Then make sure the scorer name, and body parts are the same in the config.yaml file. 

Also add for each folder a video to the `video_set` in the config.yaml file. This can also be a dummy variable, but should be e.g. 
C://somefolder.avi if the folder is called somefolder. See demo config.yaml file for proper formatting.

**Step 5**: When you are done, run ``deeplabcut.convertcsv2h5('path_to_config.yaml', scorer= 'experimenter')``

 - The scorer name must be identical to the input name for experimenter that you used when you created the project. This will automatically update "Mackenzie" to your name in the example demo notebook. 

## If you merge projects:

**Step 1**: rename the CSV files to be the target name.

**Step 2**: run and pass the target name ``deeplabcut.convertcsv2h5('path_to_config.yaml', scorer= 'experimenter')``. This will overwrite the H5 file so the data is all merged under the target name.
