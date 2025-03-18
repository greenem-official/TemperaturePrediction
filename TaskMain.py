import sys
import threading

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThreadPool
from PyQt6.QtGui import QResizeEvent, QGuiApplication
from PyQt6.QtWidgets import QApplication, QMainWindow

from App.Canvas import Canvas
from App.Data import Data
from App.states.DataState import DataState
from App.util import StylesManager, TestLogging
from App.ml.MainModel import Model


class MainWindow(QMainWindow):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data
        self.data.threadPool = QThreadPool.globalInstance()

        self.setWindowTitle("Предсказания температуры")

        # 410, 240
        screen_resolution = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(80,  # int((screen_resolution.width() - data.renderWindowSize[0]) / 2),
                         80,  # int((screen_resolution.height() - data.renderWindowSize[1]) / 2),
                         1700, # data.renderWindowSize[0],
                         900 # data.renderWindowSize[1])
                         )

        canvas = Canvas(data)
        self.setCentralWidget(canvas)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # self.data.model = Model(data)
        #
        # thread = threading.Thread(target=lambda: (
        #     self.data.model.load_data(),
        #     self.data.model.scale_data(),
        #     self.data.model.prepare_data(),
        #     self.data.model.create_model(),
        #     self.data.model.train_model(10),
        #     self.data.model.predict(),
        #     self.data.plotWidget.update_plot()
        # ))
        # thread.start()

    def resizeEvent(self, event: QResizeEvent):
        self.data.renderWindowSize = (event.size().width(), event.size().height())


def load_style(file_path):
    with open(file_path, 'r') as file:
        return file.read()


if __name__ == "__main__":
    StylesManager.init()
    data = Data(size=(1000, 600))
    data.dataState = DataState()

    # TestLogging.init_logger('log.txt')

    app = QApplication(sys.argv)
    app.setStyleSheet(StylesManager.load_style('App/data/styles/style.css'))

    window = MainWindow(data)
    window.show()
    try:
        app.exec()
    except Exception as e:
        print(f"Exception: {e}")
