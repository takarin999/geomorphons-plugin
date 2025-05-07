# Основные пути
QGIS_PYTHON_PATH = "D:/progs/QGIS/apps/qgis/python"
QGIS_SITE_PACKAGES = "D:/progs/QGIS/apps/Python39/Lib/site-packages"

import os
import sys
import numpy as np
from osgeo import gdal
from qgis.PyQt.QtWidgets import QDialog, QAction, QMessageBox, QFileDialog, QProgressDialog, QApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.uic import loadUiType
from PyQt5 import Qt
from qgis.core import QgsProject, QgsRasterLayer
from qgis.utils import iface
from numba import jit, int16, float32, prange

# Добавляем пути QGIS в sys.path
if QGIS_PYTHON_PATH not in sys.path:
    sys.path.append(QGIS_PYTHON_PATH)
if QGIS_SITE_PACKAGES not in sys.path:
    sys.path.append(QGIS_SITE_PACKAGES)

# Загрузка UI формы
FORM_CLASS, _ = loadUiType(os.path.join(
    os.path.dirname(__file__), 'ui_form.ui'
))

class GeomorphonsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dlg = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'icon.png')
        self.action = QAction(
            QIcon(icon_path),
            "Geomorphons Calculator",
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("Geomorphons", self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("Geomorphons", self.action)

    def run(self):
        if not self.dlg:
            self.dlg = GeomorphonsDialog(self.iface)
        self.dlg.show()

class GeomorphonsDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.current_max_L = 1000  # Максимальное значение L по умолчанию

        # Инициализация элементов UI
        self.pushButton.clicked.connect(self.select_file)
        self.button_run.clicked.connect(self.run_algorithm)

        # Установка дефолтных значений
        self.spinbox_L.setValue(50)
        self.spinbox_t.setValue(1.0)
        self.spinbox_L.setMaximum(self.current_max_L)
        self.template_size = 5  # Размер эталонного окна (можете заменить на spinbox для гибкости)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DEM File",
            "",
            "DEM Files (*.tif *.tiff *.asc);;All Files (*)"
        )
        if file_path:
            try:
                dataset = gdal.Open(file_path)
                if not dataset:
                    QMessageBox.warning(self, "Warning", "Selected file is not a valid DEM")
                    return

                # Проверка наличия данных
                band = dataset.GetRasterBand(1)
                if band.GetMinimum() is None or band.GetMaximum() is None:
                    band.ComputeStatistics(0)

                self.file_widget.setText(file_path)
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                self.current_max_L = min(cols, rows) // 2 - 1
                self.spinbox_L.setMaximum(max(1, self.current_max_L))
                self.spinbox_L.setValue(min(50, self.current_max_L))

                # Показ информации о DEM
                self.label_info.setText(f"DEM size: {cols}x{rows} pixels")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read DEM file: {str(e)}")
            finally:
                if 'dataset' in locals():
                    dataset = None

    def run_algorithm(self):
        # Создаем прогресс-бар
        progress = QProgressDialog("Processing DEM...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Geomorphons Calculation")
        progress.show()

        try:
            input_path = self.file_widget.text()
            if not input_path:
                QMessageBox.warning(self, "Warning", "Please select an input DEM file!")
                return

            L = int(self.spinbox_L.value())
            t = float(self.spinbox_t.value())
            template_size = self.template_size
            output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, "similarity_output.tif")

            try:
                # Проверка и удаление старого файла, если существует
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except PermissionError:
                        if QgsProject.instance().mapLayersByName("Similarity"):
                            QgsProject.instance().removeMapLayer(
                                QgsProject.instance().mapLayersByName("Similarity")[0].id()
                            )
                        try:
                            os.remove(output_path)
                        except Exception as e:
                            QMessageBox.warning(self, "Warning",
                                                f"Could not delete old file. Please close it manually and try again.\nError: {str(e)}")
                            return

                # Чтение DEM
                dataset = gdal.Open(input_path)
                if not dataset:
                    raise Exception("Failed to open DEM file")

                band = dataset.GetRasterBand(1)
                elevation = band.ReadAsArray().astype(np.float32)

                # 1. Извлечь эталонное окно (по центру DEM)
                template = self.extract_template(elevation, template_size)
                # 2. Вычислить геоморфоны для эталона
                template_geomorphons = self.calculate_geomorphons(template, L, t)
                # 3. Классификация всей карты по схожести с эталоном
                similarity_map = self.classify_by_template(elevation, template_geomorphons, template_size, L, t)
                # 4. Сохранить результат
                self.save_result(similarity_map, dataset, output_path)

                # Добавление слоя в QGIS
                if os.path.exists(output_path):
                    old_layers = QgsProject.instance().mapLayersByName("Similarity")
                    for layer in old_layers:
                        QgsProject.instance().removeMapLayer(layer.id())
                    layer = QgsRasterLayer(output_path, "Similarity")
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
                        QMessageBox.information(self, "Success", "Classification completed!")
                    else:
                        QMessageBox.warning(self, "Warning", "Invalid output layer")
                else:
                    QMessageBox.warning(self, "Warning", "Output file not created")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

            def update_progress(percent):
                progress.setValue(percent)
                QApplication.processEvents()
                return not progress.wasCanceled()

            # Модифицировать функции вычислений для поддержки прогресс-бара
            similarity_map = self.classify_by_template_with_progress(
                elevation, template_geomorphons, template_size, L, t, update_progress)

            if progress.wasCanceled():
                QMessageBox.information(self, "Info", "Processing was canceled")
                return

            # ... остальной код ...

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def extract_template(self, elevation, size):
        """Извлекает квадратное эталонное окно по центру DEM"""
        rows, cols = elevation.shape
        start_row = rows // 2 - size // 2
        start_col = cols // 2 - size // 2
        return elevation[start_row:start_row+size, start_col:start_col+size]

    def calculate_geomorphons(self, elevation, L, t):
        """Расчет геоморфонов для любого массива"""
        elevation_np = np.asarray(elevation, dtype=np.float32)
        rows, cols = elevation_np.shape
        geomorphons = np.full((rows, cols), 10, dtype=np.int16)
        if rows <= 2*L or cols <= 2*L:
            return geomorphons
        _calculate_geomorphons_numba_parallel(
            elevation_np,
            geomorphons,
            np.int16(L),
            np.float32(t)
        )
        return geomorphons

    def classify_by_template(self, elevation, template_geomorphons, template_size, L, t):
        """Классификация всей карты по схожести с эталоном с оптимизацией"""
        rows, cols = elevation.shape
        half = template_size // 2
        result = np.zeros((rows, cols), dtype=np.float32)
        template_flat = template_geomorphons.flatten()

        # Предварительная проверка размеров
        if rows < template_size or cols < template_size:
            return result

        # Векторизованная обработка
        for y in range(half, rows - half):
            for x in range(half, cols - half):
                window = elevation[y-half:y+half+1, x-half:x+half+1]
                if window.shape != (template_size, template_size):
                    continue

                # Пропуск участков с NoData значениями
                if np.any(np.isnan(window)):
                    result[y, x] = np.nan
                    continue

                geomorphons = self.calculate_geomorphons(window, L, t)
                window_flat = geomorphons.flatten()

                # Использование более сложной метрики сходства
                score = np.sum(window_flat == template_flat) / template_flat.size
                result[y, x] = score * 100  # Масштабирование 0-100

        return result

    def save_result(self, data, input_ds, output_path):
        try:
            # Проверка доступности места на диске
            required_space = data.size * 2  # Примерная оценка
            stat = os.statvfs(os.path.dirname(output_path))
            if stat.f_bavail * stat.f_bsize < required_space:
                raise Exception("Not enough disk space for output file")

            # Удаление старого файла с проверкой блокировки
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except PermissionError:
                    # Попробовать разблокировать файл через QGIS
                    layers = QgsProject.instance().mapLayersByName(os.path.basename(output_path))
                    for layer in layers:
                        QgsProject.instance().removeMapLayer(layer.id())
                    os.remove(output_path)

            # Создание выходного файла с оптимизированными параметрами
            driver = gdal.GetDriverByName("GTiff")
            options = [
                'COMPRESS=DEFLATE',
                'PREDICTOR=2',
                'ZLEVEL=9',
                'TILED=YES'
            ]
            out_ds = driver.Create(
                output_path,
                data.shape[1],
                data.shape[0],
                1,
                gdal.GDT_Int16,
                options=options
            )
            out_ds.SetGeoTransform(input_ds.GetGeoTransform())
            out_ds.SetProjection(input_ds.GetProjection())
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(data)
            out_band.SetNoDataValue(0)  # Установка значения NoData
            out_band.FlushCache()
            out_ds = None

        except Exception as e:
            raise Exception(f"Cannot write to output file: {str(e)}")

@jit(nopython=True, parallel=True)
def _calculate_geomorphons_numba_parallel(elevation, geomorphons, L, t):
    rows, cols = elevation.shape
    directions = np.array([
        [1, 0], [1, 1], [0, 1], [-1, 1],
        [-1, 0], [-1, -1], [0, -1], [1, -1]
    ], dtype=np.int16)
    for y in prange(L, rows - L):
        for x in range(L, cols - L):
            current_height = elevation[y, x]
            if np.isnan(current_height):
                geomorphons[y, x] = 0  # NoData value
                continue

            pattern = np.zeros(8, dtype=np.int16)
            for i in range(8):
                dx, dy = directions[i]
                max_diff = -np.inf
                for r in range(1, L + 1):
                    ny, nx = y + dy * r, x + dx * r
                    if 0 <= ny < rows and 0 <= nx < cols:
                        neighbor_height = elevation[ny, nx]
                        if not np.isnan(neighbor_height):
                            diff = (neighbor_height - current_height) / r
                            max_diff = max(max_diff, diff)
                if max_diff > t:
                    pattern[i] = 1
                elif max_diff < -t:
                    pattern[i] = -1
            geomorphons[y, x] = _classify_pattern_numba(pattern)

@jit(nopython=True)
def _classify_pattern_numba(pattern):
    positive = 0
    negative = 0
    for i in range(8):
        if pattern[i] == 1:
            positive += 1
        elif pattern[i] == -1:
            negative += 1
    if positive == 8:
        return 1  # Peak
    elif negative == 8:
        return 2  # Pit
    elif positive >= 6:
        return 3  # Ridge
    elif negative >= 6:
        return 4  # Valley
    elif positive >= 4:
        return 5  # Shoulder
    elif negative >= 4:
        return 6  # Hollow
    elif positive >= 2 and negative >= 2:
        return 7  # Spur
    elif (positive + negative) <= 1:
        return 8  # Flat
    else:
        return 9  # Slope
