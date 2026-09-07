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
from importlib import import_module


def main():
    try:
        import_module("deeplabcut.gui")
    except ImportError as err:
        print(err)
        return

    # if module is executed directly (i.e. `python -m deeplabcut.__init__`) launch straight into the GUI
    print("Starting GUI...")
    from deeplabcut.gui.launch_script import launch_dlc

    launch_dlc()


if __name__ == "__main__":
    main()
