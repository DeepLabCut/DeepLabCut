# Design Documentation for DeepLabCut PySide GUI

## Architecture

### Reusable components

Since there is behavior repeated in multiple places of the application, `components.py` includes some commonly used objects. 

#### Layouts

For example, 

```python
from components import _create_horizontal_layout

layout = _create_horizontal_layout()
```

will create a `QtWidgets.QHBoxLayout` object, and set the alignment, spacing, and margins to the defaults used throughout the GUI. 

#### Functional components

Since Qt does not support using the same instance of an object in various places (each QWidget can only be attached to one parent), functional components such as `VideoSelectionWidget` or `BodypartListWidget` are also defined to reduce code duplication, since this functionality is needed in several different parts of the code. 

Useful defaults are provided in the class constructor where applicable. 

### Main Window

The main window of the application inherits from `QtWidgets.QMainWindow`. This is referred to all other widgets in the program as the `root` widget and holds variables that are common in many tabs such as `shuffle` and `videotype`.

### Tabs

All the Tab classes inherit from a custom class, `components.DefaultTab` which inherits from `QtWidgets.QWidget` and implements the basic layout common in all tabs (step description, config line, browse new config file), and holds commonly variables and functionality. 

To create a tab object, one needs to provide the `root`, `parent` and `h1_description` arguments. There is only one `root` element, the Main Window. The parent refers to the current widget's direct parent, and can be the MainWindow or another tab, depending on how nested of a composition is implemented. The `h1_description` is just the information provided at the top of each tab, e.g. "DeepLabCut Step X - Label some unicorns".

### Keeping "state" of variables across tabs

The current GUI keeps the "state" of some variables synced across tabs by calling the `refresh_active_tab()` function of the `MainWindow` whenever a new tab is clicked. 

This is a rather rough solution towards having a stateful UI, and an approach where there is a `dict` storing all the desired variables would be a better approach. All the tabs can access and update the information in that `dict` directly, instead of having local copies of variables. 

A persistent-state UI (recover GUI state after closing and reopening the application) could be implemented by storing the previously mentioned dict as a JSON or YAML file in the installation path of DeepLabCut, and read the file upon opening the software.

### Concentrate global and constant values

Throughout the software, there are repeatedly used values such as supported videotypes, augmentation types ets. These are all maintained in a single file, `dlc_params.py`, and are accessed from other files:

```python

from dlc_params import DLC_Params

supported_videotypes = DLC_Params.VIDEOTYPES
```

This reduces code duplication, making the codebase more maintainable and reduces errors. Also, since these are read-only variables, they are declared as constants. 

### Logging

The GUI has been written such that any operation being performed (either clicking buttons, or running actions) is logged using Python's [`logging` module](https://docs.python.org/3/library/logging.html). 

The logging is configured in `main.py`. All the tabs use the same logger object, named `GUI`, which they access from the root widget (Main Window). This can easily be extended to a more fine-grained logging system, where each Tab can use its own logger, for an easier tracing of where each logging message is coming from. For example:

```python
from components import DefaultTab
import logging

class MyNewTab(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(MyNewTab, self).__init__(root, parent, h1_description)

        self.logger = logging.getLogger("MYLOGGER")

        ....
```

Most logging messages are sent with `INFO` level, while some messages that would not be interesting for the average user are using `DEBUG` level. Currently the logging level is set in `main.py`, but that can easily be changed to be a command line argument, or even a setting in the GUI. It can also be silenced by setting the logging level to `CRITICAL`.

### GUI Styling

A [QSS stylesheet](https://doc.qt.io/qt-5/stylesheet-reference.html)[^*] is used to define global styles for the GUI. The syntax is similar to CSS. This helps to apply the "separation of concerns" design pattern which argues that styling code and behavioral code should be separated. 

[^*]: Python specific examples [here](https://doc.qt.io/qtforpython/overviews/stylesheet-examples.html).

Benefits include the fact that all styling can be changed from a single place, instead of navigating the codebase for that, and it also reduces code duplication.  

## Improvements compared to current WxPython GUI

- Stylesheets
- Logging
- Globally used, constant values such as `videotypes=["avi", "mov", ""mp4]` is all centralized in one place only
- A basic landing page
- UI improvements such as enabling/disabling options based on current selections (e.g. in `Extract Frames` tab)
- UI improvements in video selection: 
  - show count of selected videos
  - clear video selection button
  - `filelist` is a set (cannot select same video twice)
  - if `videotype` changes, `filelist` is cleared
- Open project mini-window:
  - removed redundant "Project Loaded" info message
  - “Ok” button is automatically set in focus after selecting config file, so that you can just press “Enter” button to load (small detail, but actually quite convenient)
- Basic "state" kept for variables across tabs. Check `MainWindow.refresh_active_tab()` for what is currently kept in sync
- `config` files, `pose_config` and `inference_cfg` files are editable through a built-in Config Editor