from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtGui import QPainter, QColor

from App.Data import Data
from App.UI import UIWidget

class Canvas(QWidget):
    def __init__(self, data: Data):
        super().__init__()
        self.data = data

        self.data.canvas = self

        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        ui = UIWidget(self.data)
        self.mainLayout.addWidget(ui)
        ui.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_graphics(self):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(40, 40, 40))  # Background

        painter.end()
