class DataState:
    """
    Класс с контекстом последнего загруженного датасета (не связан напрямую с данными внутри модели)
    """
    def __init__(self):
        self.importedData = None
