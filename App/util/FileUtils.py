from PyQt6.QtWidgets import QFileDialog, QWidget


def import_csv_file_name(widget: QWidget):
    file_name, _ = QFileDialog.getOpenFileName(widget, 'Выберите CSV файл', '', 'CSV Files (*.csv)')

    return file_name
