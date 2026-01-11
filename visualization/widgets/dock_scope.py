from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                               QDialog, QFormLayout, QDoubleSpinBox, QCheckBox, QColorDialog, QFrame)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import csv
import datetime
import os

class ScopeSettingsDialog(QDialog):
    """V2.3 示波器设置 (保持 V2.2 逻辑)"""
    def __init__(self, parent=None, current_bg='k', y_range=(None, None)):
        super().__init__(parent)
        self.setWindowTitle("Oscilloscope Settings")
        self.resize(300, 200)
        self.layout = QVBoxLayout(self)

        # Appearance
        grp_app = QFormLayout()
        self.btn_color = QPushButton("Choose Color...")
        self.current_color = current_bg
        self.btn_color.setStyleSheet(f"background-color: {current_bg}; color: {'black' if current_bg in ['white', '#ffffff'] else 'white'};")
        self.btn_color.clicked.connect(self.choose_color)
        grp_app.addRow("Background:", self.btn_color)
        self.layout.addLayout(grp_app)

        # Y-Axis
        self.chk_auto_y = QCheckBox("Auto Scale Y-Axis")
        self.chk_auto_y.setChecked(y_range[0] is None)
        self.layout.addWidget(self.chk_auto_y)

        form_y = QFormLayout()
        self.spin_y_min = QDoubleSpinBox(); self.spin_y_min.setRange(-1e9, 1e9); self.spin_y_min.setValue(y_range[0] if y_range[0] else -10)
        self.spin_y_max = QDoubleSpinBox(); self.spin_y_max.setRange(-1e9, 1e9); self.spin_y_max.setValue(y_range[1] if y_range[1] else 10)
        form_y.addRow("Y Min:", self.spin_y_min)
        form_y.addRow("Y Max:", self.spin_y_max)
        self.layout.addLayout(form_y)

        self.chk_auto_y.toggled.connect(self.spin_y_min.setDisabled)
        self.chk_auto_y.toggled.connect(self.spin_y_max.setDisabled)
        self.spin_y_min.setDisabled(self.chk_auto_y.isChecked())
        self.spin_y_max.setDisabled(self.chk_auto_y.isChecked())

        btn_box = QHBoxLayout()
        btn_ok = QPushButton("OK"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_ok); btn_box.addWidget(btn_cancel)
        self.layout.addLayout(btn_box)

    def choose_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.current_color = c.name()
            self.btn_color.setStyleSheet(f"background-color: {self.current_color};")

    def get_settings(self):
        return {
            'bg_color': self.current_color,
            'auto_y': self.chk_auto_y.isChecked(),
            'y_min': self.spin_y_min.value(),
            'y_max': self.spin_y_max.value()
        }

class ScopeDockWidget(QWidget):
    """
    WOUSE V2.3 示波器
    - 全程数据记录与显示 (不截断)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # --- Toolbar ---
        btn_layout = QHBoxLayout()
        self.btn_settings = QPushButton("⚙ Settings")
        self.btn_clear = QPushButton("Clear")
        self.btn_export = QPushButton("Export")
        line = QFrame(); line.setFrameShape(QFrame.VLine); line.setFrameShadow(QFrame.Sunken)
        self.btn_auto = QPushButton("Auto Fit")
        self.btn_zoom_in = QPushButton("Zoom (+)")
        self.btn_zoom_out = QPushButton("Zoom (-)")

        btn_layout.addWidget(self.btn_settings); btn_layout.addWidget(self.btn_clear); btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(line)
        btn_layout.addWidget(self.btn_auto); btn_layout.addWidget(self.btn_zoom_in); btn_layout.addWidget(self.btn_zoom_out)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.bg_color = 'k'
        self.plot_widget.setBackground(self.bg_color)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.layout.addWidget(self.plot_widget)

        # State
        self.active_channels = []
        # [V2.3 Change] 使用 List 动态存储，绘图时转 array
        self.time_history = []
        self.data_buffers = {} # key -> list of values
        self.curves = {}
        self.data_history = []

        self.csv_file = None; self.csv_writer = None; self.current_log_path = ""
        self.y_auto = True; self.y_range = (-10, 10)

        # Connections
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_clear.clicked.connect(self.reset_session)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_auto.clicked.connect(self.on_auto_scale)
        self.btn_zoom_in.clicked.connect(self.on_zoom_in)
        self.btn_zoom_out.clicked.connect(self.on_zoom_out)

    def on_auto_scale(self):
        self.plot_widget.enableAutoRange(axis='x')
        self.plot_widget.enableAutoRange(axis='y')
        self.y_auto = True

    def on_zoom_in(self):
        self.plot_widget.plotItem.getViewBox().scaleBy((0.8, 0.8))

    def on_zoom_out(self):
        self.plot_widget.plotItem.getViewBox().scaleBy((1.25, 1.25))

    def open_settings(self):
        dlg = ScopeSettingsDialog(self, self.bg_color, (None, None) if self.y_auto else self.y_range)
        if dlg.exec():
            s = dlg.get_settings()
            self.bg_color = s['bg_color']
            self.plot_widget.setBackground(self.bg_color)
            self.y_auto = s['auto_y']
            if self.y_auto:
                self.plot_widget.enableAutoRange(axis='y')
            else:
                self.plot_widget.setYRange(s['y_min'], s['y_max'], padding=0)

    def update_channels(self, channels_config):
        self.active_channels = channels_config
        new_keys = [c['key'] for c in channels_config]

        self.plot_widget.clear()
        self.curves = {}

        # [V2.3] 重新加载完整历史数据
        times = np.array(self.time_history)

        for conf in channels_config:
            k = conf['key']
            style_map = {'Solid': Qt.SolidLine, 'Dash': Qt.DashLine, 'Dot': Qt.DotLine}
            pen = pg.mkPen(color=conf['color'], style=style_map.get(conf['style'], Qt.SolidLine), width=2)
            curve = self.plot_widget.plot(pen=pen, name=conf['name'])
            self.curves[k] = curve

            # 如果有数据，立即回填
            if k in self.data_buffers and len(self.data_buffers[k]) == len(times):
                curve.setData(times, np.array(self.data_buffers[k]))

    def update_data(self, data_frame):
        # 1. CSV Save
        if self.csv_file and self.csv_writer is None:
            fieldnames = list(data_frame.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
        if self.csv_writer: self.csv_writer.writerow(data_frame)

        # 2. Store Memory (Full History)
        self.data_history.append(data_frame)
        t = data_frame['time']
        self.time_history.append(t)

        # Update all buffers (even inactive ones, so we can switch later)
        for key, val in data_frame.items():
            if key not in self.data_buffers:
                self.data_buffers[key] = []
            self.data_buffers[key].append(val)

        # 3. Update Plots (Active only)
        # Optimization: Don't convert full list to numpy array every frame if array is huge
        # But for <100k points, modern CPUs handle it fine.
        # For simplicity in V2.3: Convert full array.
        t_arr = np.array(self.time_history)

        for conf in self.active_channels:
            key = conf['key']
            if key in self.curves and key in self.data_buffers:
                # Get full data array
                y_arr = np.array(self.data_buffers[key])
                self.curves[key].setData(t_arr, y_arr)

    def reset_session(self):
        self.data_history = []
        self.time_history = []
        self.data_buffers = {}
        self.plot_widget.clear()
        self.curves = {}
        self.update_channels(self.active_channels)
        self._init_recording()

    def _init_recording(self):
        if self.csv_file: self.csv_file.close()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join("data", f"Run_{timestamp}_LabFAST_V0.0.1")
        if not os.path.exists(folder): os.makedirs(folder)
        filename = os.path.join(folder, "full_record.csv")
        self.current_log_path = folder
        try:
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = None
        except: pass

    def export_data(self):
        if not self.data_history: return
        fn, _ = QFileDialog.getSaveFileName(self, "Export", f"WOUSE_Data.csv", "CSV (*.csv)")
        if fn:
            try:
                keys = list(self.data_history[0].keys())
                with open(fn, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.data_history)
            except: pass