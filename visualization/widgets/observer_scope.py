from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QInputDialog
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import csv


class TimeScopeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.channel_combo = QComboBox()
        self.btn_add = QPushButton("Add Channel")
        self.btn_clear = QPushButton("Clear")
        self.btn_load = QPushButton("Load History")
        controls.addWidget(QLabel("Channel:"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(self.btn_add)
        controls.addWidget(self.btn_load)
        controls.addWidget(self.btn_clear)
        controls.addStretch()
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.time_history = []
        self.data_buffers = {}
        self.curves = {}
        self.history_series = []

        self.btn_add.clicked.connect(self.add_channel)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_load.clicked.connect(self.load_history)

    def set_available_channels(self, channels: list[str]) -> None:
        self.channel_combo.clear()
        self.channel_combo.addItems(channels)

    def add_channel(self):
        key = self.channel_combo.currentText()
        if not key or key in self.curves:
            return
        curve = self.plot.plot(pen=pg.mkPen(width=2), name=key)
        self.curves[key] = curve

    def clear(self):
        self.plot.clear()
        self.curves = {}
        self.time_history = []
        self.data_buffers = {}
        self.history_series = []

    def update_data(self, data_frame: dict):
        t = data_frame.get("time")
        if t is None:
            return
        self.time_history.append(float(t))
        for key, val in data_frame.items():
            self.data_buffers.setdefault(key, []).append(val)

        t_arr = np.array(self.time_history)
        for key, curve in self.curves.items():
            y_arr = np.array(self.data_buffers.get(key, []))
            if len(y_arr) == len(t_arr):
                curve.setData(t_arr, y_arr)

    def load_history(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV (*.csv)")
        if not fn:
            return
        with open(fn, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return
        keys = list(rows[0].keys())
        key, ok = QInputDialog.getItem(self, "Select Column", "Column:", keys, 0, False)
        if not ok or not key:
            return
        t_values = [float(r.get("time", r.get("Time", i))) for i, r in enumerate(rows)]
        y_values = [float(r.get(key, 0.0)) for r in rows]
        pen = pg.mkPen(style=Qt.DashLine, width=2)
        curve = self.plot.plot(pen=pen, name=f"{key} (history)")
        curve.setData(np.array(t_values), np.array(y_values))
        self.history_series.append(curve)


class FreqScopeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.channel_combo = QComboBox()
        self.btn_add = QPushButton("Add Channel")
        self.btn_clear = QPushButton("Clear")
        controls.addWidget(QLabel("Channel:"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(self.btn_add)
        controls.addWidget(self.btn_clear)
        controls.addStretch()
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        self.time_history = []
        self.data_buffers = {}
        self.curves = {}

        self.btn_add.clicked.connect(self.add_channel)
        self.btn_clear.clicked.connect(self.clear)

    def set_available_channels(self, channels: list[str]) -> None:
        self.channel_combo.clear()
        self.channel_combo.addItems(channels)

    def add_channel(self):
        key = self.channel_combo.currentText()
        if not key or key in self.curves:
            return
        curve = self.plot.plot(pen=pg.mkPen(width=2), name=key)
        self.curves[key] = curve

    def clear(self):
        self.plot.clear()
        self.curves = {}
        self.time_history = []
        self.data_buffers = {}

    def update_data(self, data_frame: dict):
        t = data_frame.get("time")
        if t is None:
            return
        self.time_history.append(float(t))
        for key, val in data_frame.items():
            self.data_buffers.setdefault(key, []).append(val)

        if len(self.time_history) < 4:
            return

        dt = self.time_history[-1] - self.time_history[-2]
        if dt <= 0:
            return

        for key, curve in self.curves.items():
            y = np.array(self.data_buffers.get(key, []))
            if len(y) < 4:
                continue
            n = len(y)
            y_detrend = y - np.mean(y)
            fft_vals = np.fft.rfft(y_detrend)
            freq = np.fft.rfftfreq(n, d=dt)
            amp = np.abs(fft_vals) / n
            curve.setData(freq, amp)
