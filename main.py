import PySide2.QtWidgets as QtWidgets
import sys
from PySide2.QtGui import QIcon

from MainWindow import MainWindow


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('./assets/logo.png'))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
