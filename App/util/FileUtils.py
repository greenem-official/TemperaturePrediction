from PyQt6.QtWidgets import QFileDialog, QWidget

"""
Утилиты диалоговых окон импорта файлов
"""

def import_file_name(widget: QWidget):
    file_name, _ = QFileDialog.getOpenFileName(widget, 'Выберите CSV или TXT файл', '', 'CSV Files (*.csv);;TXT Files (*.txt)')
    return file_name

def import_file_name_csv(widget: QWidget):
    file_name, _ = QFileDialog.getOpenFileName(widget, 'Выберите CSV файл', '', 'CSV Files (*.csv)')  # ;;All Files (*)
    return file_name
