from PyQt6.QtGui import QPainter

class Data:
    debug = False
    renderWindowSize: tuple[int, int] = None

    def __init__(self, size: tuple[int, int]):
        self.renderWindowSize = size
        self.model = None
        self.plotWidget = None
        self.lossWidget = None
        self.logWidget = None
        self.plotRangeWidget = None
        self.modelTypeWidget = None
        self.dataState = None
        self.threadPool = None
        self.validation_split = 0.2
        self.lossYMaxDisplay = 1.0
        self.graph_actual_data_visible = True
        self.graph_predicted_data_visible = True
