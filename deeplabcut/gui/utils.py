from PySide2 import QtCore


class Worker(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        self.func()
        self.finished.emit()


def move_to_separate_thread(func):
    thread = QtCore.QThread()
    worker = Worker(func)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.finished.connect(thread.deleteLater)
    return worker, thread


def is_latest_deeplabcut_version():
    import json
    import urllib.request
    from deeplabcut import VERSION

    url = 'https://pypi.org/pypi/deeplabcut/json'
    contents = urllib.request.urlopen(url).read()
    latest_version = json.loads(contents)['info']['version']
    return VERSION == latest_version, latest_version
