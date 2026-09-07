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
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import logging
import sys

import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from deeplabcut.gui.gui_assets import get_style_qss, icon_from_resource, pixmap_from_resource

logger = logging.getLogger(__name__)


def launch_dlc():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(icon_from_resource("logo.png"))
    screen_size = app.screens()[0].size()
    pixmap = pixmap_from_resource("welcome.png").scaledToWidth(int(0.7 * screen_size.width()), Qt.SmoothTransformation)
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()

    app.setStyleSheet(get_style_qss())
    try:
        # Always pass `palette=`: the zero-argument load_stylesheet() and the
        # load_stylesheet_<binding>() helpers overwrite os.environ["QT_API"],
        # which desynchronises matplotlib from the binding qtpy loaded.
        app.setStyleSheet(qdarkstyle.load_stylesheet(palette=DarkPalette))
    except Exception:
        logger.warning("Could not load the qdarkstyle stylesheet; keeping the bundled style.qss.", exc_info=True)

    # Set up a logger and add an stdout handler.
    # A single logger can have many handlers:
    # https://docs.python.org/3/howto/logging.html#handler-basic
    # TODO Dump to log file instead
    # logger = logging.getLogger("GUI")
    # logger.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler(stream=sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    # )
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    from deeplabcut.gui.window import MainWindow

    window = MainWindow(app)
    window.receiver.start()
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_dlc()
