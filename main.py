import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'
import logging

import PySide2.QtWidgets as QtWidgets
import qdarkstyle
from PySide2.QtGui import QIcon

from MainWindow import MainWindow


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('./assets/logo.png'))

    stylefile = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "style.qss"
    )
    with open(stylefile, "r") as f:
        app.setStyleSheet(f.read())

    dark_stylesheet = qdarkstyle.load_stylesheet_pyside2()
    app.setStyleSheet(dark_stylesheet)

    # Set up a logger and add an stdout handler. 
    # A single logger can have many handlers: 
    # https://docs.python.org/3/howto/logging.html#handler-basic
    logger = logging.getLogger("GUI")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    window = MainWindow(app)
    window.receiver.start()
    window.showMaximized()
    sys.exit(app.exec_())
