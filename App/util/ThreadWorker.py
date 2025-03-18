from PyQt6.QtCore import QRunnable, pyqtSignal, QObject


class WorkerSignals(QObject):
    finished = pyqtSignal(object)


class ThreadWorker(QRunnable):
    """
    Собственная обёртка вокруг QRunnable для асинхронного выполнения задач
    """
    def __init__(self, runFunc, *args):
        super().__init__()
        self.runFunc = runFunc
        self.args = args
        self.signals = WorkerSignals()

    def run(self):
        result = self.runFunc(*self.args)
        self.signals.finished.emit(result)
