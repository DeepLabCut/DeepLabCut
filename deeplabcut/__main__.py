#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

try:
    import PySide6

    lite = False
except ModuleNotFoundError:
    lite = True

# if module is executed directly (i.e. `python -m deeplabcut.__init__`) launch straight into the GUI
if not lite:
    print("Starting GUI...")
    from deeplabcut.gui.launch_script import launch_dlc

    launch_dlc()
else:
    print(
        "You installed DLC lite, thus GUI's cannot be used. If you need GUI support please: pip install 'deeplabcut[gui]''"
    )
