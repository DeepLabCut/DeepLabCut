import PyQt5.QtWidgets as QtWidgets
import sys
from PyQt5.QtGui import QIcon

from MainWindow import MainWindow


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('./assets/logo.png'))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
