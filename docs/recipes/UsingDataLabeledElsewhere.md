# Using data labeled in other software in DeepLabCut

You may have annotation data in different formats, yet want to use the DeepLabCut pipeline. In this case, you need to convert the data to our format. Simply, you can format your data in an excel sheet (.csv file) or pandas array (.h5 file).

Here is a guide to do this via the ".csv" route: (the pandas array route is identical, just format the pandas array in the same way).

1. Create a project as described in our [single animal](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html) and [multi-animal](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html) user guides. <br><br>

2. Edit the `config.yaml` file to include the body part names. Please take care that spelling, spacing, and capitalization are **identical** to the "labeled data body part names". <br><br>

3. Inspect the excel formatted sheet (.csv) from our demo project. [[this file](https://github.com/AlexEMG/DeepLabCut/blob/master/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.csv)] <br><br>

4. Edit the .csv file such that it contains the X, Y pixel coordinates, the body part names, the scorer name as well as the relative path to the image: e.g. `/labeled-data/somefolder/img17.jpg`, and make sure the scorer name, and body parts are the same in the `config.yaml` file.

    For each images folder added in the `labeled-data` folder, you need to add a video path to the `video_sets` key in the `config.yaml` file. This can also be a dummy variable (i.e. the video doesn't need to exist in the given path), but should have the same name as the folder in `labeled-data`, e.g. `C:/somepath/some-folder.avi` if the folder is called `some-folder`. Here is an example of proper formatting:
    ```yaml
    #For the given folders in labeled-data: 
    #   labeled-data
    #       ├── some-folder
    #       └── some-other-folder
    #
    #The expected entries in config.yaml are:
    video-sets:
        /dummy/path/that/doesnt/necessarily/exist/some-folder.avi:
            crop: 0, 640, 0, 480
        /videos/some-other-folder.avi:
            crop: 0, 640, 0, 480
    ```
    <br><br>

5. Run `deeplabcut.convertcsv2h5('path/to/config.yaml', scorer= 'experimenter')`. The **scorer name must be identical** to the input name for experimenter that you used when you created the project. In our [example demo notebook](https://colab.research.google.com/github/DeepLabCut/DeepLabCut/blob/master/examples/COLAB/COLAB_DEMO_mouse_openfield.ipynb), `scorer: Mackenzie` will automatically update to your name.
