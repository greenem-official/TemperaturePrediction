import os
import random

from App.Data import Data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd

from keras.src.optimizers import Adam
from keras.src.layers import LSTM, Dense, GRU, SimpleRNN
from keras import Sequential, Input
from keras._tf_keras.keras.utils import set_random_seed
# from keras._tf_keras.keras.utils import split_dataset

from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split

from App.ml.util.Scalers import MinMaxScaler

seed = 10
set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Model:
    def __init__(self, data: Data):
        self.data = data

        self.inputData = None
        self.scaled_data = None
        self.scaler = None

        self.X = None
        self.Y = None
        self.look_back = None
        self.model = None
        self.predictions = None
        self.prediction_range = 0

    def load_data(self):
        # import os
        # current_directory = os.getcwd()
        # self.inputData = pd.read_csv(current_directory + os.sep + 'monthlyMeteoData.csv', parse_dates=['time'], index_col='time')
        self.inputData = self.data.dataState.importedData  # Может быть сценарий, где это ещё не инициализировано

    def scale_data(self):
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.inputData[['temperature']])  # можно поменять на номер

    def __prepare_data(self, data, look_back=12):
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
        look_back = 12
        self.__prepare_data(self.scaled_data, look_back)
        self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))

    def create_model(self, modelType=None):
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
        self.model.fit(self.X, self.Y, epochs=epochs, batch_size=1, verbose=2)

    def predict_future(self, steps=12):
        result = []

        prev_values = self.Y[-self.look_back:].reshape(-1)  # 12 последних значений, поначалу только исходных

        for _ in range(steps):
            pred_x = prev_values[-self.look_back:].reshape(1, self.look_back, 1)  # reshape

            pred_y = self.model.predict(pred_x)  # Предсказание
            result.append(pred_y[0, 0])  # Конечный результат

            prev_values = np.append(prev_values, pred_y[0, 0])  # Удаление старого значения и добавление нового

        self.predictions = self.scaler.inverse_transform(np.array(result).reshape(-1, 1)) # Возвращение в исходный масштаб

    def predict(self, months=12):
        self.prediction_range = months
        self.predict_future(steps=self.prediction_range)

    def plot(self, ax: Axes):
        actual_from = self.data.plotRangeWidget.slider.getValue()

        predictions_x = pd.date_range(
            start=self.inputData['time'].iloc[-1] + pd.DateOffset(months=1),
            periods=self.prediction_range,
            freq='ME'
        )

        # print(predictions_x)
        # print(self.inputData['time'])
        # print([self.inputData['time'].iloc[-1], predictions_x[0]])

        # Построение фактических данных
        ax.plot(
            self.inputData['time'][actual_from:],  # [12 * 8:],
            self.inputData['temperature'][actual_from:],  # [12 * 8:],
            label='Фактические',
            marker='o'
        )

        # Линия между последним реальным и первым предсказанным значением
        ax.plot(
            [self.inputData['time'].iloc[-1], predictions_x[0]],
            [self.inputData['temperature'].iloc[-1], self.predictions[0].item()],
            color='gray',
            linestyle='--'
        )

        # Построение предсказанных данных
        ax.plot(
            predictions_x,
            self.predictions,
            label='Предсказанные',
            marker='o'
        )

        # Сетка
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.grid(True, which='major', linestyle='-', linewidth=1)

        # Отметки
        ax.set_xticks(self.inputData['time'][actual_from:])
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Поворот подписей
        plt.setp(ax.get_xticklabels(), rotation=10, ha="right")

        # Прочее
        ax.legend()
