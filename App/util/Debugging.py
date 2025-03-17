from PyQt6.QtGui import QColor, QPainter, QMouseEvent
from PyQt6.QtWidgets import QWidget


class DebuggableQWidget(QWidget):
    bgColor = None

    def __init__(self, data, colorName):
        super().__init__()
        self.data = data
        self.colorName = colorName
        self.setAcceptDrops(False)
        self.ignoreMouse = True

    def paintEvent(self, a0, painter=None):
        if not self.data.debug:
            if painter is None:
                painter = QPainter(self)
            if self.bgColor is not None:
                painter.fillRect(self.rect(), self.bgColor)
        else:
            if painter is None:
                painter = QPainter(self)
            if self.colorName is not None:
                painter.fillRect(self.rect(), color_map[self.colorName])

    def mousePressEvent(self, event: QMouseEvent):
        if self.ignoreMouse:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.ignoreMouse:
            event.accept()
        else:
            super().mouseMoveEvent(event)


__opacity = 'cc'
__opacityLess = '66'

color_map = {
    "debugHBox": QColor(f"#{__opacityLess}76efb8"),
    "debugVbox": QColor(f"#{__opacity}1aa3ff"),
    "debugCornerElement": QColor(f"#{__opacity}944dff"),
    "debugAdvancedNumberWidget": QColor(f"#{__opacity}ff4d4d"),
    "darkWidgetBg": QColor(20, 20, 20, 80),
}