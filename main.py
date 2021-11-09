# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#import PyQt5
import numpy as np
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
#from PyQt5.QtWinExtras import QtWin


from MainApp import *
from MainWindow import *
import deeplabcut
#try:
#    myappid = 'mycompany.myproduct.subproduct.version'
#    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
#except ImportError:
#    pass



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('C:/Users/User/PycharmProjects/pictures/logo.png'))

    #app.setStyle('Windows')
    #app.setStyleSheet("Windows")

    window = MainWindow()

    window.show()
    sys.exit(app.exec_())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
