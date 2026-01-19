from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                               QPushButton, QListWidget, QListWidgetItem, QComboBox,
                               QGroupBox, QFrame, QFileDialog)
from PySide6.QtCore import Qt, QTimer
import pyqtgraph as pg
import numpy as np
import csv
import os


class ObserverDockWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.available_channels = []
        self.history_sets = {}
        self.live_time = []
        self.live_buffers = {}

        ctrl = QHBoxLayout()
        self.spin_time = QSpinBox(); self.spin_time.setRange(0, 8); self.spin_time.setValue(2)
        self.spin_freq = QSpinBox(); self.spin_freq.setRange(0, 8); self.spin_freq.setValue(1)
        self.btn_load = QPushButton("Load History")
        self.btn_clear = QPushButton("Clear History")
        ctrl.addWidget(QLabel("Time scopes:")); ctrl.addWidget(self.spin_time)
        ctrl.addWidget(QLabel("Freq scopes:")); ctrl.addWidget(self.spin_freq)
        ctrl.addWidget(self.btn_load); ctrl.addWidget(self.btn_clear)
        ctrl.addStretch()
        self.layout.addLayout(ctrl)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(line)

        self.time_container = QVBoxLayout()
        self.freq_container = QVBoxLayout()
        self.layout.addLayout(self.time_container)
        self.layout.addLayout(self.freq_container)
        self.layout.addStretch()

        self.time_scopes = []
        self.freq_scopes = []

        self.spin_time.valueChanged.connect(self.rebuild_scopes)
        self.spin_freq.valueChanged.connect(self.rebuild_scopes)
        self.btn_load.clicked.connect(self.load_history)
        self.btn_clear.clicked.connect(self.clear_history)

        self.rebuild_scopes()

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(200)
        self._update_timer.timeout.connect(self.refresh_scopes)
        self._update_timer.start()

    def rebuild_scopes(self):
        self._clear_layout(self.time_container)
        self._clear_layout(self.freq_container)
        self.time_scopes = []
        self.freq_scopes = []

        for i in range(self.spin_time.value()):
            scope = TimeScopeWidget(f"Time Scope {i + 1}")
            scope.set_available_channels(self.available_channels)
            scope.set_sources(self._source_names())
            self.time_container.addWidget(scope)
            self.time_scopes.append(scope)

        for i in range(self.spin_freq.value()):
            scope = FreqScopeWidget(f"Freq Scope {i + 1}")
            scope.set_available_channels(self.available_channels)
            scope.set_sources(self._source_names())
            self.freq_container.addWidget(scope)
            self.freq_scopes.append(scope)

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def set_available_channels(self, channels):
        self.available_channels = channels
        for scope in self.time_scopes + self.freq_scopes:
            scope.set_available_channels(channels)

    def update_live_buffers(self, time_history, data_buffers):
        self.live_time = time_history
        self.live_buffers = data_buffers

    def refresh_scopes(self):
        if not self.live_time:
            return
        for scope in self.time_scopes:
            scope.update_plot(self.live_time, self.live_buffers, self.history_sets)
        for scope in self.freq_scopes:
            scope.update_plot(self.live_time, self.live_buffers, self.history_sets)

    def load_history(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load History", "data", "CSV (*.csv)")
        for path in files:
            try:
                name = os.path.basename(path)
                data = self._read_csv(path)
                if data:
                    self.history_sets[name] = data
            except Exception:
                continue
        self._refresh_sources()

    def clear_history(self):
        self.history_sets = {}
        self._refresh_sources()

    def _refresh_sources(self):
        sources = self._source_names()
        for scope in self.time_scopes + self.freq_scopes:
            scope.set_sources(sources)

    def _source_names(self):
        return ["Live"] + list(self.history_sets.keys())

    def _read_csv(self, path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        keys = reader.fieldnames or []
        data = {k: [] for k in keys}
        for r in rows:
            for k in keys:
                try:
                    data[k].append(float(r.get(k, 0.0)))
                except Exception:
                    data[k].append(0.0)
        return data


class ScopeBaseWidget(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(title))
        self.combo_channel = QComboBox()
        head.addWidget(QLabel("Channel:"))
        head.addWidget(self.combo_channel)
        head.addStretch()
        layout.addLayout(head)

        self.list_sources = QListWidget()
        self.list_sources.setMaximumHeight(70)
        layout.addWidget(self.list_sources)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

        self.curves = {}

    def set_available_channels(self, channels):
        current = self.combo_channel.currentText()
        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()
        self.combo_channel.addItems(channels)
        if current in channels:
            self.combo_channel.setCurrentText(current)
        self.combo_channel.blockSignals(False)

    def set_sources(self, sources):
        current = {self.list_sources.item(i).text(): self.list_sources.item(i).checkState()
                   for i in range(self.list_sources.count())}
        self.list_sources.clear()
        for s in sources:
            item = QListWidgetItem(s)
            state = current.get(s, Qt.Checked if s == "Live" else Qt.Unchecked)
            item.setCheckState(state)
            self.list_sources.addItem(item)

    def _selected_sources(self):
        items = []
        for i in range(self.list_sources.count()):
            item = self.list_sources.item(i)
            if item.checkState() == Qt.Checked:
                items.append(item.text())
        return items


class TimeScopeWidget(ScopeBaseWidget):
    def update_plot(self, live_time, live_buffers, history_sets):
        key = self.combo_channel.currentText()
        if not key:
            return
        sources = self._selected_sources()
        self.plot.clear()
        self.curves = {}
        for s in sources:
            if s == "Live":
                if key in live_buffers:
                    self.plot.plot(np.array(live_time), np.array(live_buffers[key]), pen=pg.mkPen(width=2), name="Live")
            else:
                data = history_sets.get(s, {})
                if key in data:
                    t = data.get("time", data.get("Time", list(range(len(data[key])))))
                    self.plot.plot(np.array(t), np.array(data[key]), pen=pg.mkPen(width=1), name=s)


class FreqScopeWidget(ScopeBaseWidget):
    def update_plot(self, live_time, live_buffers, history_sets):
        key = self.combo_channel.currentText()
        if not key:
            return
        sources = self._selected_sources()
        self.plot.clear()
        self.curves = {}
        for s in sources:
            if s == "Live":
                t = np.array(live_time)
                y = np.array(live_buffers.get(key, []))
                self._plot_fft(t, y, "Live")
            else:
                data = history_sets.get(s, {})
                t = np.array(data.get("time", data.get("Time", [])))
                y = np.array(data.get(key, []))
                self._plot_fft(t, y, s)

    def _plot_fft(self, t, y, name):
        if len(t) < 3 or len(y) < 3:
            return
        dt = np.mean(np.diff(t))
        if dt <= 0:
            return
        y = y - np.mean(y)
        n = len(y)
        freq = np.fft.rfftfreq(n, d=dt)
        amp = np.abs(np.fft.rfft(y)) / n
        self.plot.plot(freq, amp, pen=pg.mkPen(width=1), name=name)
