
from MainWindow import *
import deeplabcut
#try:
#    myappid = 'mycompany.myproduct.subproduct.version'
#    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
#except ImportError:
#    pass



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon('./pictures/logo.png'))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
