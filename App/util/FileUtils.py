from PyQt6.QtWidgets import QFileDialog, QWidget


def import_file_name(widget: QWidget):
    file_name, _ = QFileDialog.getOpenFileName(widget, 'Выберите CSV или TXT файл', '', 'CSV Files (*.csv);;TXT Files (*.txt)')
    return file_name
