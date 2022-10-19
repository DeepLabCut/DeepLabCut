"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
import logging

import PySide2.QtWidgets as QtWidgets
import qdarkstyle
from deeplabcut.gui import BASE_DIR
from PySide2.QtCore import Qt
from PySide2.QtGui import QIcon, QPixmap


def launch_dlc():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join(BASE_DIR, 'assets', 'logo.png')))
    desktop = app.desktop()
    pixmap = (
        QPixmap(os.path.join(BASE_DIR, 'assets', 'welcome.png'))
        .scaledToWidth(int(0.7 * desktop.width()), Qt.SmoothTransformation)
    )
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()

    stylefile = os.path.join(BASE_DIR, "style.qss")
    with open(stylefile, "r") as f:
        app.setStyleSheet(f.read())

    dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
    app.setStyleSheet(dark_stylesheet)

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
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_dlc()
