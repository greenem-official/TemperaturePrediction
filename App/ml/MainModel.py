import os
import random

from App.Data import Data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd

# from keras.src.optimizers import Adam
# from keras.src.layers import LSTM, Dense, GRU, SimpleRNN
# from keras import Sequential, Input
# from keras._tf_keras.keras.utils import set_random_seed
# from keras.src.saving import load_model

from keras.api.optimizers import Adam
from keras.api.layers import LSTM, Dense, GRU, SimpleRNN
from keras.api import Sequential, Input
from keras.api.utils import set_random_seed
from keras.api.saving import load_model

from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
import matplotlib.dates as mdates

from App.ml.util.Scalers import MinMaxScaler

seed = 10
set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


"""
Функция из Keras, специально вынесенная в этот файл для того, чтобы не импортировать Keras в неподходящих местах
и не ставить дополнительных переменных окружения. Это также позволяет упростить поиск проблем с Tensorflow.
"""
def load_model_from_save(file_name):
    load_model(file_name)

class Model:
    """
    Основной класс модели, который использует функционал Keras.
    В функциях этого класса подготавливаются данные, создаётся модель, делаются предсказания и рисуются основной и график потерь.
    """
    def __init__(self, data: Data):
        """
        Инициализация
        :param data: Переменная общей информации
        """
        self.data = data

        self.inputData = None
        self.scaled_data = None
        self.scaler = None

        self.X = None
        self.Y = None

        # self.X_train = None
        # self.X_test = None
        # self.Y_train = None
        # self.Y_test = None
        # self.accuracy_predictions = None

        self.loss_values = None

        self.look_back = None
        self.model = None
        self.predictions = None
        self.prediction_range = 0

    def load_data(self):
        """
        Функция загрузки в объект модели информации о последнем загруженном пользователем датасете
        """
        # import os
        # current_directory = os.getcwd()
        # self.inputData = pd.read_csv(current_directory + os.sep + 'monthlyMeteoData.csv', parse_dates=['time'], index_col='time')
        self.inputData = self.data.dataState.importedData  # Может быть сценарий, где это ещё не инициализировано

    def scale_data(self):
        """
        Функция масштабирования данных перед использованием
        """
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.inputData[['temperature']])  # можно поменять на номер

    def __prepare_data(self, data, look_back=12):
        """
        Функция подготовки X и Y numpy массивов, созданных из единого столбца температур
        Y становится просто температурами, а X - массивом предыдущих температур длиной look_back, идущих перед конкретным y.
        Графический интерфейс не предоставляет возможности менять значение look_back.
        """
        self.X, self.Y = [], []
        self.look_back = look_back
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = data

        for i in range(len(data_values) - self.look_back):
            self.X.append(data_values[i:(i + self.look_back), 0])
            self.Y.append(data_values[i + self.look_back, 0])

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def prepare_data(self):
        """
        Функция подготовки X и Y numpy массивов для обучения модели и предсказания значений
        """
        look_back = 12
        self.__prepare_data(self.scaled_data, look_back)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

    def create_model(self, modelType=None):
        """
        Инициализация внутреннего аттрибута model, хранящего настроенный объект Keras модели
        :param modelType: Тип модели в формате строки
        """
        requestedType = modelType
        if requestedType is None:
            requestedType = self.data.modelTypeWidget.modelType.getValue()

        if requestedType == 'LSTM':
            main_layer = LSTM(units=50, activation='relu')
        elif requestedType == 'GRU':
            main_layer = GRU(units=50, activation='relu')
        elif requestedType == 'SimpleRNN':
            main_layer = SimpleRNN(units=50, activation='relu')
        else:
            main_layer = LSTM(units=50, activation='relu')

        self.model = Sequential()
        self.model.add(Input(shape=(self.look_back, 1)))
        self.model.add(main_layer)  # LSTM GRU SimpleRNN
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=Adam())

        return self

    def train_model(self, epochs):
        """
        Обучение модели на X и y с указанным в слайдере validation_split
        :param epochs: Количество эпох обучения. Чем больше эпох, тем лучше предсказания. Оценить оптимальное значение помогает график потерь.
        """

        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
        #                                                     self.Y,
        #                                                     test_size=0.2,
        #                                                     shuffle=True,
        #                                                     # random_state=seed
        #                                                     )

        validation_split = self.data.validation_split
        if validation_split is None:
            validation_split = 0.2

        history = self.model.fit(self.X,
                                 self.Y,
                                 epochs=epochs,
                                 batch_size=1,
                                 validation_split=validation_split,
                                 verbose=2
                                 )

        self.loss_values = history.history['loss']

        # self.accuracy_predictions = self.model.predict(self.X_test)
        # self.

    def predict_future(self, steps=12):
        """
        Функция предсказания массива длиной steps следующих значений, идущих после данных в загруженном датасете
        :param steps: Количество предсказываемых значений
        """
        result = []

        prev_values = self.Y[-self.look_back:].reshape(-1)  # 12 последних значений, поначалу только исходных

        for _ in range(steps):
            pred_x = prev_values[-self.look_back:].reshape(1, self.look_back, 1)  # reshape

            pred_y = self.model.predict(pred_x)  # Предсказание
            result.append(pred_y[0, 0])  # Конечный результат

            prev_values = np.append(prev_values, pred_y[0, 0])  # Удаление старого значения и добавление нового

        self.predictions = self.scaler.inverse_transform(np.array(result).reshape(-1, 1)) # Возвращение в исходный масштаб

    def predict(self, months=12):
        """
        Функция предсказания значений
        :param months: Количество месяцев для предсказания (по умолчанию 12)
        """
        if self.data.dataState.importedData is not None and self.Y is None:
            self.load_data()
            self.scale_data()
            self.prepare_data()

        self.prediction_range = months
        self.predict_future(steps=self.prediction_range)

    def plot(self, ax: Axes):
        """
        Функция рисования основного графика с исходными и/или предсказанными данными
        :param ax: Объект Axes, на котором функции позволяется рисовать
        """
        actual_from = self.data.plotRangeWidget.slider.getValue()

        indexColumn = []

        if 'time' in self.inputData:
            indexColumn = self.inputData['time']

            predictions_x = pd.date_range(
                start=indexColumn.iloc[-1] + pd.DateOffset(months=1),
                periods=self.prediction_range,
                freq='ME'
            )
        elif 'index' in self.inputData:
            indexColumn = self.inputData['index']

            start_value = indexColumn.iloc[-1] + 1
            predictions_x = np.arange(
                start=start_value,
                stop=start_value + self.prediction_range,
                step=1
            )

        # print(predictions_x)
        # print(self.inputData['time'])
        # print([self.inputData['time'].iloc[-1], predictions_x[0]])

        if self.data.graph_actual_data_visible:
            # Построение фактических данных
            ax.plot(
                indexColumn[actual_from:],  # [12 * 8:],
                self.inputData['temperature'][actual_from:],  # [12 * 8:],
                label='Исходные',
                marker='o',
                color='#1f77b4'
            )

        if self.data.graph_actual_data_visible and self.data.graph_predicted_data_visible:
            # Линия между последним реальным и первым предсказанным значением
            ax.plot(
                [indexColumn.iloc[-1], predictions_x[0]],
                [self.inputData['temperature'].iloc[-1], self.predictions[0].item()],
                color='gray',
                linestyle='--'
            )

        if self.data.graph_predicted_data_visible:
            # Построение предсказанных данных
            ax.plot(
                predictions_x,
                self.predictions,
                label='Предсказанные',
                marker='o',
                color='#ff7f0e'
            )

        # Сетка
        if 'time' in self.inputData:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        elif 'index' in self.inputData:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.grid(True, which='major', linestyle='-', linewidth=1)

        # Отметки
        if self.data.graph_actual_data_visible:
            ax.set_xticks(indexColumn[actual_from:])

        if 'time' in self.inputData:
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif 'index' in self.inputData:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        # Поворот подписей
        plt.setp(ax.get_xticklabels(), rotation=10, ha="right")

        # Прочее
        if self.data.graph_actual_data_visible or self.data.graph_predicted_data_visible:
            ax.legend(loc='upper left')

    def plotLoss(self, ax: Axes):
        """
        Функция рисования графика функции потерь, т.е. зависимости точности модели от количества эпох
        :param ax: Объект Axes, на котором функции позволяется рисовать
        """
        if self.loss_values is None:
            return

        if self.data.lossYMaxDisplay is not None and self.data.lossYMaxDisplay > 0:
            # Если нужен масштаб не до 0, то раскомментировать bottom
            ax.set_ylim(top=self.data.lossYMaxDisplay * max(self.loss_values)) #, bottom=min(self.loss_values))

        if self.data.lossXMinDisplay is not None:
            ax.set_xlim(left=self.data.lossXMinDisplay * len(self.loss_values), right=len(self.loss_values))

        ax.plot(range(1, len(self.loss_values) + 1), self.loss_values, marker='o', linestyle='-')
        # ax.title('')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Значение потерь')
        # ax.grid(True)
