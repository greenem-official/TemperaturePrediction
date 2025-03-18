from enum import Enum

"""
Файл для загруки CSS стилей приложения
"""

def load_style(file_path):
    with open(file_path, 'r') as file:
        return file.read()


class StyleType(Enum):
    NormalButton = 1,
    SectionTitle = 2,
    SimpleText = 3,

def getStyle(style):
    return styles[style]

styles = {}

def init():
    pass
    styles[StyleType.NormalButton] = load_style('App/data/styles/NormalButton.css')
    styles[StyleType.SectionTitle] = load_style('App/data/styles/SectionTitle.css')
    styles[StyleType.SimpleText] = load_style('App/data/styles/SimpleText.css')
