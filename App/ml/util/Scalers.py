import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.data_min_ = 0
        self.data_max_ = 0

    def fit_transform(self, data):
        self.data_min_ = np.min(data)
        self.data_max_ = np.max(data)
        scaled_data = (data - self.data_min_) / (self.data_max_ - self.data_min_)

        return scaled_data

    def inverse_transform(self, scaled_data):
        return scaled_data * (self.data_max_ - self.data_min_) + self.data_min_