import os
from qgis.PyQt import uic

def load_ui(ui_file):
    """Правильная загрузка UI файлов с виджетами QGIS"""
    ui_path = os.path.join(os.path.dirname(__file__), ui_file)
    return uic.loadUiType(ui_path)[0]