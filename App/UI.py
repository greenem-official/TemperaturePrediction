import sys
import threading

import pandas as pd
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QLabel, QSlider, QHBoxLayout, QSpacerItem, \
    QSizePolicy, QSpinBox, QTextEdit, QWidget, QLineEdit, QComboBox, QCheckBox
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout

from App.util import FileUtils, TestLogging
from App.util.Output import RedirectOutput
from App.util.ThreadWorker import ThreadWorker

# print(plt.style.available)
# plt.style.use('dark_background')
plt.style.use('dark_custom.mplstyle')

from App.Data import Data
from App.util.StylesManager import StyleType, getStyle
from App.util.Debugging import color_map, DebuggableQWidget
from App.ml.MainModel import Model


class AdvancedCheckboxWidget(DebuggableQWidget):
    def __init__(self, data: Data, name, default, onValueChange=None):
        super().__init__(data, 'debugAdvancedNumberWidget')
        self.data = data

        self.funcOnRelease = onValueChange
        self.name = name

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.checkbox = QCheckBox(self)
        self.checkbox.setChecked(default)

        self.checkbox.stateChanged.connect(self.update_value)
        if self.funcOnRelease is not None:
            self.checkbox.stateChanged.connect(self.funcOnRelease)

        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.result_label.setStyleSheet(self.result_label.styleSheet() + "font-size: 16px;")
        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.result_label)

        # self.setStyleSheet(self.styleSheet() + 'border-bottom: 2px solid white;')

        self.update_value(self.checkbox.isChecked())
        # self.show()
        self.update()

    def getValue(self):
        return self.checkbox.isChecked()

    def update_value(self, value):
        self.result_label.setText(f'{self.name}')

class AdvancedSpinboxWidget(DebuggableQWidget):
    def __init__(self, data: Data, name, range, default, onValueChange=None):
        super().__init__(data, 'debugAdvancedNumberWidget')
        self.data = data

        self.funcOnRelease = onValueChange
        self.name = name

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.spinbox = QSpinBox(self)
        self.spinbox.setRange(range[0], range[1])
        self.spinbox.setValue(default)

        self.spinbox.valueChanged.connect(self.update_value)
        if self.funcOnRelease is not None:
            self.spinbox.valueChanged.connect(self.funcOnRelease)

        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.spinbox)

        self.update_value(self.spinbox.value())
        # self.show()
        self.update()

    def getValue(self):
        return self.spinbox.value()

    def update_value(self, value):
        self.result_label.setText(f'{self.name}')


class AdvancedSliderWidget(DebuggableQWidget):
    def __init__(self, data: Data, name, range, default, onValueChange=None, displayValueFunc=None, preciseUpdate=False, verticalSlider=False):
        super().__init__(data, 'debugAdvancedNumberWidget')
        self.data = data

        self.funcOnRelease = onValueChange
        self.scaleValueFunc = displayValueFunc
        self.name = name
        self.verticalSlider = verticalSlider

        if not self.verticalSlider:
            self.layout = QVBoxLayout()
        else:
            self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.slider = QSlider(Qt.Orientation.Vertical if self.verticalSlider else Qt.Orientation.Horizontal, self)
        self.slider.setRange(range[0], range[1])
        self.slider.setValue(default)

        selectedSignal = self.slider.sliderReleased
        if preciseUpdate:
            selectedSignal = self.slider.valueChanged
        selectedSignal.connect(self.update_value)
        if self.funcOnRelease is not None:
            selectedSignal.connect(self.funcOnRelease)

        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.slider)

        self.update_value()
        # self.show()
        self.update()

    def getValue(self):
        return self.slider.value()

    def getScaledValue(self):
        value = self.getValue()
        if self.scaleValueFunc is not None:
            return self.scaleValueFunc(value)
        else:
            return value

    def update_value(self):
        valueText = ' '
        if self.scaleValueFunc is not None:
            valueText += '{0:.2f}'.format(self.scaleValueFunc(self.getValue()))
        self.result_label.setText(f'{self.name}' + valueText)  # : {value}')


class AdvancedComboBoxWidget(DebuggableQWidget):
    def __init__(self, data: Data, name, options, default, onValueChange=None):
        super().__init__(data, 'debugAdvancedNumberWidget')
        self.data = data

        self.funcOnRelease = onValueChange
        self.name = name

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(options)
        self.combo_box.setCurrentIndex(default)

        self.combo_box.currentIndexChanged.connect(self.update_value)
        if self.funcOnRelease is not None:
            self.combo_box.currentIndexChanged.connect(self.funcOnRelease)

        self.result_label = QLabel('', self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.combo_box)

        self.update_value()
        # self.show()
        self.update()

    def getValue(self):
        return self.combo_box.currentIndex()

    def update_value(self):
        self.result_label.setText(f'{self.name}')  # : {value}')


class MatplotlibWidget(DebuggableQWidget):
    def __init__(self, data: Data, onCanvasDraw, subplotsAdjust=None):
        super().__init__(data, 'debugAdvancedNumberWidget')
        # super().__init__(parent)
        self.onCanvasDraw = onCanvasDraw

        self.data = data

        # Создаем фигуру и оси
        self.figure = Figure()
        if subplotsAdjust is not None:
            self.figure.subplots_adjust(**subplotsAdjust)

        self.ax = self.figure.add_subplot(111)

        # Создаем холст для отображения графика
        self.canvas = FigureCanvas(self.figure)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 0, 15, 0)
        self.setLayout(self.layout)

        self.layout.addWidget(self.canvas)

    def update_plot(self):
        # with plt.style.context('dark_background'):
        self.ax.clear()
        # self.ax.plot(x, y)
        if self.onCanvasDraw is not None:
            self.onCanvasDraw(self.ax)
        self.canvas.draw()


class LogWidget(DebuggableQWidget):
    def __init__(self, data):
        super().__init__(data, 'debugAdvancedNumberWidget')
        self.data = data
        self.data.logWidget = self

        self.max_characters = 100000

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.log_text)

    def append_text(self, message):
        updated_text = self._process_backspace(message)
        updated_text = self._apply_character_limit(updated_text)

        self.log_text.setPlainText(updated_text)
        self._scroll_to_bottom()

    def _process_backspace(self, message):
        current_text = self.log_text.toPlainText()
        new_text = list(current_text)

        for char in message:
            if char == '\x08':  # backspace
                if new_text:
                    new_text.pop()
            else:
                new_text.append(char)

        return ''.join(new_text)

    def _apply_character_limit(self, text):
        if len(text) > self.max_characters:
            while len(text) > self.max_characters:
                first_newline_index = text.find('\n')
                if first_newline_index == -1:
                    text = text[len(text) - self.max_characters:]
                else:
                    text = text[first_newline_index + 1:]
        return text

    def _scroll_to_bottom(self):
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())


class ModelChoiceWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data
        self.data.modelTypeWidget = self
        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        visualTitle = QLabel('Тип модели')
        visualTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        visualTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(visualTitle)

        self.modelType = AdvancedComboBoxWidget(data=data, name='Тип модели', options=('LSTM', 'GRU', 'SimpleRNN'), default=0,
                                                      onValueChange=None)
        layout.addWidget(self.modelType)


class DataInputWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data
        self.importWorker = None

        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        visualTitle = QLabel('Данные')
        visualTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        visualTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(visualTitle)

        self.importButton = QPushButton('Импорт...')
        self.importButton.clicked.connect(self.onImportButton)

        # layout.addWidget(QLabel('Датасет'))
        layout.addWidget(self.importButton)

    # Выполнение загрузки файла откреплено от главного потока
    def onImportButton(self):
        # print('Импорт данных...\n')
        file_name = FileUtils.import_csv_file_name(self)
        if file_name == '':
            return

        self.importWorker = ThreadWorker(self.import_data, file_name)
        self.importWorker.signals.finished.connect(self.on_import_finished)

        self.data.threadPool.start(self.importWorker)

    def import_data(self, file_name):
        print("Загрузка файла...\n")

        df = pd.read_csv(file_name)
        df.columns = ['time', 'temperature']
        df['time'] = pd.to_datetime(df['time'])

        df = df.resample('ME', on='time').mean()
        df = df.reset_index()

        return df

    def on_import_finished(self, df):
        QTimer.singleShot(0, lambda: self.process_imported_data(df))

    def process_imported_data(self, df):
        self.data.dataState.importedData = df

        print("Данные (вычисленно среднее по месяцам):")
        print(df)

        print("\nДанные успешно загружены\n")


class ModelTrainingWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data

        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        visualTitle = QLabel('Обучение')
        visualTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        visualTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(visualTitle)

        self.epochsWidget = AdvancedSpinboxWidget(data=data, name='Эпохи', range=(0, 10000), default=10,
                                                  onValueChange=None
                                                  )
        self.validationSplitWidget = AdvancedSliderWidget(data=data, name='Доля валидации', range=(0, 20), default=4,
                                                          onValueChange=self.on_validation_split_value_change,
                                                          displayValueFunc=self.scale_validation_value,
                                                          preciseUpdate=True
                                                          )

        self.trainButton = QPushButton('Обучить')
        self.trainButton.clicked.connect(self.onTrainButton)

        layout.addWidget(self.epochsWidget)
        layout.addWidget(self.validationSplitWidget)
        layout.addWidget(self.trainButton)

    def scale_validation_value(self, value):
        return value / 20

    def on_validation_split_value_change(self):
        self.data.validation_split = self.validationSplitWidget.getScaledValue()

    def onTrainButton(self):
        self.data.model = Model(self.data)

        worker = ThreadWorker(self.train_model)
        worker.signals.finished.connect(lambda: (
            self.data.lossWidget.update_plot(),
        ))

        self.data.threadPool.start(worker)

    def train_model(self):
        if self.data.dataState.importedData is None:
            return

        print('Обучение...\n')

        self.data.model.load_data(),
        self.data.model.scale_data(),
        self.data.model.prepare_data(),
        self.data.model.create_model(),
        self.data.model.train_model(epochs=self.epochsWidget.getValue())

        print('\nМодель успешно обучена\n')


class ModelPredictionWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data

        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        visualTitle = QLabel('Предсказание')
        visualTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        visualTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(visualTitle)

        self.monthsRangeWidget = AdvancedSpinboxWidget(data=data, name='Месяцев', range=(0, 1000), default=12, onValueChange=None)

        self.importButton = QPushButton('Предсказать')
        self.importButton.clicked.connect(self.onPredictButton)

        layout.addWidget(self.monthsRangeWidget)
        layout.addWidget(self.importButton)

    def onPredictButton(self):
        if self.data.model is None or self.data.model.model is None:
            return

        print('Предсказание значений...\n')

        worker = ThreadWorker(lambda: (
            self.data.model.predict(months=self.monthsRangeWidget.getValue()),
        ))
        worker.signals.finished.connect(lambda: (
            self.data.plotWidget.update_plot(),
            print('\nПредсказания успешно созданы\n')
        ))

        self.data.threadPool.start(worker)


class ModelSavesWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data

        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        visualTitle = QLabel('Сохранение модели')
        visualTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        visualTitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(visualTitle)

        self.save_button = QPushButton('Сохранить')
        self.save_button.clicked.connect(self.on_save_button)

        self.load_button = QPushButton('Загрузить')
        self.load_button.clicked.connect(self.on_load_button)

        layout.addWidget(self.save_button)
        layout.addWidget(self.load_button)

    def on_save_button(self):
        pass

    def on_load_button(self):
        pass


class PlotRange(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugCornerElement')
        self.data = data
        self.data.plotRangeWidget = self

        self.bgColor = color_map['darkWidgetBg']

        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel('Диапазон графика')
        title.setStyleSheet(getStyle(StyleType.SectionTitle))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider = AdvancedSliderWidget(data=data, name='Начиная с недели', range=(0, 100), default=0, onValueChange=self.on_slider_change)

        layout.addWidget(title)
        layout.addWidget(self.slider)

    def on_slider_change(self):
        self.data.plotWidget.update_plot()

    def update_sliders(self):
        self.slider.slider.setRange(0, self.data.dataState.importedData.shape[0])


class HorizSpacer(QSpacerItem):
    def __init__(self, width):
        super().__init__(width, 0, QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)


class VertSpacer(QSpacerItem):
    def __init__(self, height):
        super().__init__(0, height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)


class TopWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugVbox')
        self.data = data

        spacer_width = 20

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        self.setLayout(layout)

        modelChoiceWidget = ModelChoiceWidget(data)
        importWidget = DataInputWidget(data)
        trainWidget = ModelTrainingWidget(data)
        predictionWidget = ModelPredictionWidget(data)
        savesWidget = ModelSavesWidget(data)

        layout.addWidget(modelChoiceWidget)
        layout.addItem(HorizSpacer(spacer_width))

        layout.addWidget(importWidget)
        layout.addItem(HorizSpacer(spacer_width))

        layout.addWidget(trainWidget)
        layout.addItem(HorizSpacer(spacer_width))

        layout.addWidget(predictionWidget)
        layout.addItem(HorizSpacer(spacer_width))

        layout.addWidget(savesWidget)
        layout.addItem(HorizSpacer(spacer_width))

        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # modelChoiceWidget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)


class GraphRangeWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugVbox')
        self.data = data

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        sliderLayout = QHBoxLayout()
        sliderLayout.setContentsMargins(0, 0, 0, 0)

        sliderLayout.addWidget(PlotRange(data))

        self.actualCheckbox = AdvancedCheckboxWidget(data, name='Исходные данные', default=True, onValueChange=self.on_actual_checkbox_changed)
        self.predictedCheckbox = AdvancedCheckboxWidget(data, name='Предсказанные данные', default=True, onValueChange=self.on_predicted_checkbox_changed)

        layout.addWidget(self.actualCheckbox)
        layout.addWidget(self.predictedCheckbox)
        layout.addWidget(self.predictedCheckbox)

        layout.addLayout(sliderLayout)

    def on_actual_checkbox_changed(self, state):
        self.data.graph_actual_data_visible = state
        self.data.plotWidget.update_plot()

    def on_predicted_checkbox_changed(self, state):
        self.data.graph_predicted_data_visible = state
        self.data.plotWidget.update_plot()

class MainGraphWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugVbox')
        self.data = data

        self.bgColor = color_map['graphCombinedWidgetBg']

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        self.setLayout(layout)

        mainPlotWidget = MatplotlibWidget(data, onCanvasDraw=self.onMainCanvasDraw, subplotsAdjust=({'bottom': 0.2}))
        mainPlotWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        bottomWidget = GraphRangeWidget(data)
        bottomWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        title = QLabel('Значения')
        title.setStyleSheet(getStyle(StyleType.SectionTitle))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(mainPlotWidget)

        layout.addItem(VertSpacer(10))

        layout.addWidget(bottomWidget)

        self.data.plotWidget = mainPlotWidget

    def onMainCanvasDraw(self, ax):
        if self.data.model is not None:
            self.data.model.plot(ax)


class LossGraphWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugVbox')
        self.data = data

        self.bgColor = color_map['graphCombinedWidgetBg']

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        self.setLayout(layout)

        rightLayout = QVBoxLayout()
        rightLayout.setContentsMargins(0, 10, 0, 10)

        self.lossGraph = MatplotlibWidget(data, onCanvasDraw=self.onLossCanvasDraw, subplotsAdjust=({'left': 0.35, 'bottom': 0.2}))

        title = QLabel('Функция потерь')
        title.setStyleSheet(getStyle(StyleType.SectionTitle))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        rightLayout.addWidget(title)

        rightLayout.addItem(VertSpacer(10))

        rightLayout.addWidget(self.lossGraph)

        self.lossYWidget = AdvancedSliderWidget(data=data, name='Y', range=(0, 100), default=100,
                                                onValueChange=self.on_loss_value_change,
                                                displayValueFunc=self.scale_loss_value,
                                                verticalSlider=True
                                                )

        layout.addWidget(self.lossYWidget)
        layout.addLayout(rightLayout)

        self.data.lossWidget = self.lossGraph

    def onLossCanvasDraw(self, ax):
        if self.data.model is not None:
            self.data.model.plotLoss(ax)

    def scale_loss_value(self, value):
        return value / 100

    def on_loss_value_change(self):
        self.data.lossYMaxDisplay = self.lossYWidget.getScaledValue()
        self.lossGraph.update_plot()

class CombinedBottomGraphsWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugVbox')
        self.data = data

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        mainGraph = MainGraphWidget(data)
        lossGraph = LossGraphWidget(data)

        layout.addWidget(mainGraph, 3)
        layout.addWidget(lossGraph, 1)


class UIWidget(DebuggableQWidget):
    def __init__(self, data: Data):
        super().__init__(data, 'debugHBox')
        self.data = data
        self.ignoreMouse = False

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        consoleTitle = QLabel('Консоль')
        consoleTitle.setStyleSheet(getStyle(StyleType.SectionTitle))
        consoleTitle.setAlignment(Qt.AlignmentFlag.AlignLeft)
        consoleTitle.setContentsMargins(10, 0, 0, 0)
        layout.addWidget(consoleTitle)

        logWidget = LogWidget(data)
        # mainPlotWidget = MatplotlibWidget(data, onCanvasDraw=self.onMainCanvasDraw)
        # lossPlotWidget = MatplotlibWidget(data, onCanvasDraw=self.onLossCanvasDraw)

        topWidget = TopWidget(data)
        # bottomWidget = GraphRangeWidget(data)

        spacer = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        logWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        logWidget.setMaximumHeight(150)
        # mainPlotWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        topWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # bottomWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(logWidget)
        layout.addWidget(topWidget)

        # layout.addItem(spacer)
        # layout.addWidget(mainPlotWidget)
        #
        # layout.addWidget(bottomWidget)

        layout.addWidget(CombinedBottomGraphsWidget(data))

        self.redirect_output = RedirectOutput()
        self.redirect_output.text_written.connect(self.data.logWidget.append_text)
        sys.stdout = self.redirect_output
        print('Здесь будет отображаться консольный вывод\n')

        self.data.plotWidget.update_plot()
        self.data.lossWidget.update_plot()

    def isPosInsideOfRect(self, pos: QPoint, rect: QRect):
        return rect.contains(pos)
