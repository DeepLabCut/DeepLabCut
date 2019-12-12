"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os

# if module is executed direcyly (i.e. `python -m deeplabcut.__init__`) launch straight into the GUI
if os.environ.get("DLClight", default=False) == False:
    print("Starting GUI...")
    import deeplabcut
    deeplabcut.launch_dlc()
else:
    print("You are in DLClight mode, GUI cannot be started.")
