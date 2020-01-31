"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os

guistate=os.environ.get("DLClight", default="False")

# if module is executed directly (i.e. `python -m deeplabcut.__init__`) launch straight into the GUI
if guistate=="False": # or not guistate:
    print("Starting GUI...")
    import deeplabcut
    deeplabcut.launch_dlc()
else:
    print("You are in DLClight mode, GUI cannot be started.")
