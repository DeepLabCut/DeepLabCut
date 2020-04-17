### The DeepLabCut Prediction Plugin System

The DeepLabCut plugin system allows for the rapid development of new 
pose prediction algorithms. These plugins accept raw probability data
(TrackingData) from DeepLabCut and return the expected locations of 
body parts (Poses). Below is a simple guide that describes how to 
create plugins and use them in DeepLabCut:

### Default Plugins

DeepLabCut currently comes with 3 "default" plugins:
 - "argmax": The default plugin, which makes predictions by simply finding the 
   point with the highest probability within the probability frame.
 - "plotterargmax": Makes predictions in the same exact way as "argmax",
   but also outputs the probability maps to a video for visualization.
 - "fast_plotterargmax": A much faster implementation of "plotterargmax", but
   offers less customization and lacks a 3D mode.  

### Using Plugins

Given an existing project and trained network, a prediction plugin can be 
applied to a video by identifying the plugin in the to `analyze_videos()` 
function call:

```python
deeplabcut.analyze_videos(config_path, ["/fullpath/project/videos/"], predictor="plugin_name")
```

To list all the currently installed plugins, call the list_predictor_plugins()
from the python or ipython console:

```python
# Make sure you have imported DeepLabCut first.
import deeplabcut
deeplabcut.list_predictor_plugins()
```
The above method will load all the currently available plugins, and
display their names followed by their descriptions. `


##### Plugin Settings:

Some plugins will come with settings that can be configured from the
config.yaml. To list the settings of plugins,  the 
`deeplabcut.get_predictor_settings` method can be used, as below:

```python
# Get settings for 1 plugin:
deeplabcut.get_predictor_settings("plugin_name")
# Get settings for more than 1 plugin:
deeplabcut.get_predictor_settings(["plugin_name1", "plugin_name2", ...])
# Get settings for all plugins currently available:
deeplabcut.get_predictor_settings()
```

These settings can be configured in the config.yaml:

```yaml
predictors:
  "plugin_name":
    "setting_name": Value
  "another_plugin_name":
    "another_setting": Value
```

For example, the "plotterargmax" plugin offers several settings 
which allow the user to modify the appearance of the probability maps in 
the final rendered video. An example configuration in the config.yaml 
is shown below:

```yaml
predictors:
  plotterargmax:
    # Change the name of the output video.
    video_name: "coolvideo.mp4"
    # Change the video codec used...
    codec: "H264"
    # Using log scale for more sensitive color maps.
    use_log_scale: True
    # Changing the matplotlib colormap used while plotting.
    colormap: "inferno"
    # Use 3D rather then 2D projection.
    3d_projection: True
```

### Creating a Plugin

It is possible to create your own prediction plugins (also referred to 
as "predictors"). Since predictors are dynamically loaded, these plugins
can be installed and rewritten on the fly. For a class to be loaded 
as a prediction plugin, there are 2 requirements:

 - The class extends the `deeplabcut.pose_estimation_tensorflow.nnet.processing.Predictor`
   abstract class and overrides all of the abstract methods. 
 - The class is located in the 'pose_estimation_tensorflow/nnet/predictors' 
   folder of the DeepLabCut installation, or any sub-folder of this 
   folder.

##### Creating An Example Plugin

To begin, you will need to find where your DeepLabCut install is located, 
and set it as your working directory. One way to do this is to open 
the ipython console and execute the commands below:

```python
import deeplabcut
dlcpath = list(deeplabcut.__path__)[0]
# To print the install directory...
print(dlcpath)
# To cd into the install directory...
cd $dlcpath
```

Once in the DeepLabCut install directory, move to the 'pose_estimation_tensorflow/nnet/predictors'
folder, and create a file for the plugin:

```sh
cd "pose_estimation_tensorflow/nnet/predictors"
echo '' > example_plugin.py
```

Now, open the 'example_plugin.py' in your favorite text editor, and 
copy and paste the code below:

```python
# For types in methods
from typing import Union, List, Tuple, Any, Callable, Dict

# Plugin base class and API classes...
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, TrackingData, Pose


class PrintAndPredict(Predictor):
    """
    A simple example plugin which prints frames to the command line...
    """
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int, settings: None, video_metadata: Dict[str, Any]):
        """ 
        Method for setting up the predictor plugin, configuring settings, 
        and initializing data structures
        """
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_outputs = num_outputs


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """
        Per-Frame Processing... Receives a batch of probability frames
        and expects poses or none in return if algorithm needs more 
        frames to make the prediction.
        """
        # Print data to console...
        print(scmap.get_source_map())
        # Return prediction by selecting maximums...
        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))

    def on_end(self, pbar) -> Union[None, Pose]:
        """
        Post-Processing, receives a progress bar, expects any poses not
        returned in on_frames, we already processed all the frames so
        we return None here...
        """
        return None

    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        """ Plugin offers no settings """
        return None

    @classmethod
    def get_name(cls) -> str:
        """ Plugin Name goes here """
        return "example_plugin"

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        """ No test methods... """
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        """ We support num_outputs > 1 in this plugin... """
        return True
```

Now you can run your plugin! (See "Using Plugins", above).

##### Key Predictor Methods
Of the abstract methods overridden by a predictor plugin, there are 3
that are key to the workflow of a plugin.

 - `__init__`: Where a plugin should configure settings, setup needed 
               data structures, and initialize anything else that is 
               needed.
 - `on_frames`: This is executed every time a batch of probability 
                frames is received from the DLC neural network. This is
                where your plugin should do the majority of it's processing,
                and return frames if the algorithm used requires no post-processing.
 - `on_end`: This is executed after all frames have been processed, and
             should be used by your plugin if it needs to perform 
             post processing or can only make predictions after receiving
             all of the frames. If no post processing is needed, rather
             than returning poses, just return None.

##### Testing for Plugins

The api also allows for automated testing of plugins by a rather 
primitive testing API. A plugin specifies its test methods by 
returning a list of callable from the `get_test` method. These methods 
should accept no arguments and return a boolean and two strings. The boolean
determines if the test was successful, and the two strings represent 
the expected and actual results. In order to run plugin tests, simply
use the `deeplabcut.test_predictor_plugin` method as shown below:

```python
# Test 1 predictor plugin:
deeplabcut.test_predictor_plugin("predictor_name")
# Test several predictor plugins:
deeplabcut.test_predictor_plugin(["predictor_name1", "predictor_name2", ...])
# Test all predictor plugins:
deeplabcut.test_predictor_plugin()
# Slow down output by waiting for user input between tests:
deeplabcut.test_predictor_plugin(interactive=True)
```
