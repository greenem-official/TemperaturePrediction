import re

from PyQt6.QtCore import QObject, pyqtSignal
# from ansi2html import Ansi2HTMLConverter


class RedirectOutput(QObject):
    text_written = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # self.converter = Ansi2HTMLConverter()

    def remove_ansi_escape_sequences(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _process_carriage_returns(self, text):
        lines = text.split('\r')
        if len(lines) > 1:
            return lines[-1]
        return text

    def write(self, text):
        # html_text = self.converter.convert(text, full=False)

        processed_text = text
        processed_text = self.remove_ansi_escape_sequences(processed_text)
        # processed_text = self._process_carriage_returns(processed_text)

        self.text_written.emit(processed_text)

    def flush(self):
        pass
