
from PyQt5.QtWidgets import QMainWindow

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)

        self.createStatusBar()

    def createStatusBar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        QtWidgets.QMainWindow.setStatusBar(status)
