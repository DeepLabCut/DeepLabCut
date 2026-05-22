---
deeplabcut:
  last_metadata_updated: '2026-05-22'
  last_verified: '2026-05-22'
  verified_for: 3.0.0rc14
  ignore: false
---

(file:recipe-importing-data)=

# Importing annotated data from elsewhere

## Using data labeled elsewhere

Some users may have annotation data in different formats, yet want to use the DLC pipeline. In this case, you need to convert the data to our format. Simply, you can format your data in an excel sheet (.csv file) or pandas array (.h5 file).

Here is a guide to do this via the ".csv" route: (the pandas array route is identical, just format the pandas array in the same way).

1. Create a project

1. Edit the `config.yaml` file to include the body part names, please take care that spelling, spacing, and capitalization are IDENTICAL to the "labeled data body part names".

1. Please inspect the excel formatted sheet (.csv) from our [demo project](https://github.com/DeepLabCut/DeepLabCut/tree/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1)

   - e.g. see [this file](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.csv)

1. Edit the .csv file such that it contains the X, Y pixel coordinates, the body part names, the scorer name as well as the relative path to the image: e.g. /labeled-data/somefolder/img017.jpg
   Then make sure the scorer name, and body parts are the same in the config.yaml file.
   Also add for each folder a video to the `video_set` in the config.yaml file.
   This can also be a dummy variable, but should be e.g. `C://somefolder.avi` if the folder is called `somefolder`. See demo config.yaml file for proper formatting.

1. When you are done, run `deeplabcut.convertcsv2h5('path_to_config.yaml', scorer= 'experimenter')`

- The scorer name must be identical to the input name for experimenter that you used when you created the project. This will automatically update "Mackenzie" to your name in the example demo notebook.

## Merging multiple datasets

1. Rename the CSV files to be the target name.
1. Run and pass the target name `deeplabcut.convertcsv2h5('path_to_config.yaml', scorer= 'experimenter')`. This will overwrite the H5 file so the data is all merged under the target name.
