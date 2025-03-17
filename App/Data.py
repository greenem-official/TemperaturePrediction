from PyQt6.QtGui import QPainter

class Data:
    debug = False
    renderWindowSize: tuple[int, int] = None

    def __init__(self, size: tuple[int, int]):
        self.renderWindowSize = size
        self.model = None
        self.plotWidget = None
        self.logWidget = None
        self.plotRangeWidget = None
        self.modelTypeWidget = None
        self.dataState = None
        self.threadPool = None
