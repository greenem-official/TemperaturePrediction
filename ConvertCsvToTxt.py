import sys

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QMainWindow, QPushButton, QApplication
from matplotlib import pyplot as plt

from App.util import FileUtils, StylesManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Конвертирование форматов")
        self.setGeometry(80, 80, 800, 500)

        self.csvToTxtBtn = QPushButton("CSV в TXT")
        self.csvToTxtBtn.clicked.connect(self.csvToTxtFunc)

        self.csvToTxtBtn.setFixedWidth(200)
        self.csvToTxtBtn.setFixedHeight(100)

        self.layout().addWidget(self.csvToTxtBtn)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def csvToTxtFunc(self):
        file_name = FileUtils.import_file_name_csv(self)
        if file_name == '':
            return
        if not file_name.endswith('.csv'):
            print('Некорректный формат!')
            return

        df = import_csv_data(file_name)
        file_main_name = ''.join(file_name.split('.')[:-1])

        save_to_txt(df, file_main_name + '.txt')


    def closeEvent(self, event):
        plt.close('all')
        event.accept()


def import_csv_data(file_name):
    df = pd.read_csv(file_name)
    df.columns = ['time', 'temperature']
    df['time'] = pd.to_datetime(df['time'])

    df = df.resample('ME', on='time').mean()
    df = df.reset_index()

    return df

def save_to_txt(df, file_name):
    df['temperature'].to_csv(file_name, encoding='utf-8', index=False, header=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(StylesManager.load_style('App/data/styles/style.css'))

    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec())
    except Exception as e:
        print(f"Exception: {e}")
